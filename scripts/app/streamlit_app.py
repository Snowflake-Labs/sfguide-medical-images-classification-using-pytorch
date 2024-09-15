# Import necessary packages
import streamlit as st
from snowflake.snowpark.context import get_active_session
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, udf, sproc
from snowflake.ml.registry import Registry
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import pandas as pd
import random

st.set_page_config(layout="wide")

def detect_pneumonia_spcs(session: Session,
                         database_name: str,
                         schema_name: str,
                         model_name: str,
                         model_version: str,
                         deployment_model_name: str,
                         infer_stagelib: str,
                         image_limit: int) -> str:
    
    import torch.nn as nn
    from snowflake.ml.registry import Registry

    def test_loop(modelversion, testdata, loss_fn):
        print('*' * 5 + 'Testing Started' + '*' * 5)
        
        full_pred, full_lab = [], []
        TestLoss, TestAcc = 0.0, 0.0
        
        for data, target in testdata:
            
            # Ensure data has correct shape: (3, 224, 224)
            image_tensor = data[0].unsqueeze(0)  # Add batch dimension
            
            image_tensor = torch.nn.functional.interpolate(image_tensor, size=(224, 224))  # Resize to correct size
            image_tensor = image_tensor.to(dtype=torch.float32)  # Ensure data type matches the model signature
    
            
            output = modelversion.run([image_tensor.numpy()])  # Convert tensor to numpy array for Snowflake
            
            # Print output for debugging
            print("Model Output:", output)    
            
            if isinstance(output, list) and len(output) > 0:
                df = pd.DataFrame(output[0])
                if df.empty:
                    st.write("Output DataFrame is empty")
                else:
                    df = df.rename(columns={i: f'output_{i}' for i in range(df.shape[1])})
                    output = torch.tensor(df.values)
                    loss = loss_fn(output, target)
    
                    _, pred = torch.max(output.data, 1)
                    st.write(f'Predicted values (batch): {pred.tolist()}')
                    TestLoss += loss.item() * data.size(0)
                    TestAcc += torch.sum(pred == target.data)
                    torch.cuda.empty_cache()
                    full_pred += pred.tolist()
                    full_lab += target.data.tolist()
                    
            else:
                st.write(" ")
                return [], []
        st.write(f'All Predicted values: {full_pred}')
        st.write(f'All Actual labels: {full_lab}')
        TestLoss = TestLoss / len(testdata.dataset)
        TestAcc = TestAcc / len(testdata.dataset)
        
        print(f'Loss: {TestLoss} Accuracy: {TestAcc}%')
        return full_pred, full_lab

    db = database_name
    schema = schema_name

    native_registry = Registry(session=session, database_name=db, schema_name=schema)
    modelversion = native_registry.get_model(model_name).version(model_version)

    directory = os.getcwd()
    file_list = []

    # Get list of all files in both folders
    session.sql(f'''ls {infer_stagelib}/chest_xray/test/NORMAL''').collect()
    normal_rows = session.sql(f'''select substr("name", position('/', "name")+1) file from table(RESULT_SCAN(LAST_QUERY_ID()))''').collect()
    
    session.sql(f'''ls {infer_stagelib}/chest_xray/test/PNEUMONIA''').collect()
    pneumonia_rows = session.sql(f'''select substr("name", position('/', "name")+1) file from table(RESULT_SCAN(LAST_QUERY_ID()))''').collect()

    # Combine both lists
    all_rows = normal_rows + pneumonia_rows

    # Randomly select the number of images specified by the user
    selected_rows = random.sample(all_rows, min(image_limit, len(all_rows)))

    for row in selected_rows:
        folder = "NORMAL" if row in normal_rows else "PNEUMONIA"
        file_list.append(session.sql(f"SELECT GET_PRESIGNED_URL({infer_stagelib}, '{row[0]}', 3600);").collect()[0][0])
        session.file.get(infer_stagelib + "/" + row[0], directory + f"/tmp/test/{folder}")

    # Ensure your transformations are correct
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    test_folder = directory + "/tmp/test/"
    testset = datasets.ImageFolder(test_folder, transform=transform)
    test_dl = DataLoader(testset, batch_size=32)

    loss_fn = nn.CrossEntropyLoss()
    pred, lab = test_loop(modelversion, test_dl, loss_fn)
    
    df_pred = pd.DataFrame(pred, columns=['pred'])
    df_pred['label'] = lab
    df_pred['url'] = file_list
    
    # Write to Snowflake table
    session.create_dataframe(df_pred).write.mode("overwrite").save_as_table(model_name + "_metrics")

    return f"Inference completed, check Snowflake table {model_name}_metrics for output"

def run_inference(_sp_session, app_db, app_sch, model_name, 
                  model_version, deployment_model_name, infer_stagelib, image_count):
    session = _sp_session

    return_text = detect_pneumonia_spcs(session, app_db, 
                                        app_sch, 
                                        model_name,
                                        model_version,
                                        deployment_model_name, 
                                        infer_stagelib,
                                        image_count)
    if return_text == 'Success':
        df_out = _sp_session.sql(f"select * from {app_db}.{app_sch}.{model_name}_metrics").to_pandas()
        return df_out
    else:
        return pd.DataFrame(list(return_text))

# Streamlit UI setup
st.title('Detect PNEUMONIA')

session = get_active_session()

app_role = session.sql("SELECT CURRENT_ROLE()").collect()[0][0]

app_db = (session.get_current_database()).lower().replace('"', "")
app_sch = (session.get_current_schema()).lower().replace('"', "")

model_name = "DICOM_pytorch_model_multigpu"
model_version = "V1"
deployment_model_name = "DICOM_pytorch_model_multigpu_v1"
infer_stagelib = "@data_stage"
images_df = None

# Single input for number of images
c1, c2, c3 = st.columns(3)
with c1:
    image_sample_count = st.number_input('Please select number of images', value=1, min_value=1, max_value=10)
with c3:
    if st.button('Run Inference'):
        
        df = run_inference(session, app_db, app_sch, model_name, model_version, 
                    deployment_model_name, infer_stagelib, image_sample_count)
        images_df = session.table("{0}.{1}.{2}_metrics".format(app_db, app_sch, model_name ))
        
        
try:
    for image in images_df.to_local_iterator():
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(image['url'])
            with col2:
                if image['label'] == 0:
                    st.write('Original Diagnosis: NORMAL')
                else:
                    st.write('Original Diagnosis: PNEUMONIA')
            with col3:
                if image['pred'] == 0:
                    st.write('Model Predicts: NORMAL')
                else:
                    st.write('Model Predicts: PNEUMONIA')
except Exception as e:
    st.info("Please run the inference")
