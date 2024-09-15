import os, sys
import numpy as np 
import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import logging

import common_utils as U

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from PIL import *
from tqdm import tqdm

from torchvision import datasets, transforms
from torch.utils.data import  DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split

from snowflake.snowpark import Session
from snowflake.snowpark.version import VERSION
from snowflake.snowpark.types import StructType, StructField, FloatType, StringType, IntegerType, List
import snowflake.snowpark.functions as Fn

from snowflake.ml.registry import Registry
from snowflake.ml._internal.utils import identifier
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("modeltrainreg")

# Create Snowflake Session object
# connection_parameters = json.load(open('sflk_connection.json'))
# session = Session.builder.configs(U.get_connection_info).create()
session = U.connect_to_snowflake()
session.sql_simplifier_enabled = True
config = U.get_config('.')


role_name=session.sql("SELECT CURRENT_ROLE()")

def logmessage(message):
    session.sql(f'''
    create table training_log if not exists(
    timest timestamp_ltz default current_timestamp(),
    log_message string
    ) ''').collect()
        
    logger.info(message)    
    session.sql(f'''insert into training_log (log_message) values ('{message}')''').collect()

try:
    

    logger.info("starting!")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using {} device".format(device))


    model_name = "DICOM_pytorch_model_multigpu"
    snowflake_model_version = "v1"

    stage_name = 'data_stage'

    # Get list of all files in the stage
    stage_name = '@data_stage'
    local_dir = '/tmp'

    # Define the subfolders to be fetched from the stage
    subfolders = [
        'chest_xray/train/PNEUMONIA',
        'chest_xray/train/NORMAL',
        'chest_xray/test/PNEUMONIA',
        'chest_xray/test/NORMAL'
    ]

    # Ensure the local directory exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Fetch files from each subfolder and place them in the corresponding local subfolder
    for subfolder in subfolders:
        # Create the corresponding local subfolder path
        local_subfolder = os.path.join(local_dir, subfolder)
        if not os.path.exists(local_subfolder):
            os.makedirs(local_subfolder)
        
        # Fetch the files from the Snowflake stage
        session.file.get(f"{stage_name}/{subfolder}/", local_subfolder)

    # Optionally, print the first few files to confirm
    for root, dirs, files in os.walk(local_dir):
        print(f"Folder: {root}")
        print(f"Subfolders: {dirs}")
        print(f"Files: {files[:3]}")  # Print first 3 files
        logger.info("Iam here33")

    
    logger.info("Iam here4")
    train_folder= '/tmp/chest_xray/train/'
    test_folder = '/tmp/chest_xray/test/'
    
    data_transforms = {
        'train': {
            'dataset1': transforms.Compose([transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomGrayscale(),
                transforms.RandomAffine(translate=(0.05,0.05), degrees=0),
                transforms.ToTensor()
            ]),

            'dataset2' : transforms.Compose([transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomGrayscale(),
                transforms.RandomAffine(translate=(0.1,0.05), degrees=10),
                transforms.ToTensor()

            ]),
            'dataset3' : transforms.Compose([transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomGrayscale(p=1),
                transforms.RandomAffine(translate=(0.08,0.1), degrees=15),
                transforms.ToTensor()
            ]),
        },
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]),
    }



    logger.info('starting image transform 1')
    dataset1 = datasets.ImageFolder(train_folder, 
                        transform=data_transforms['train']['dataset1'])



    logger.info('starting image transform 2')
    dataset2 = datasets.ImageFolder(train_folder, 
                        transform=data_transforms['train']['dataset2'])



    logger.info('starting image transform 3')
    dataset3 = datasets.ImageFolder(train_folder, 
                        transform=data_transforms['train']['dataset3'])



    logger.info('starting train_test_split 1')
    norm1, _ = train_test_split(dataset2, test_size= 0.05, shuffle=False)


    logger.info('starting train_test_split 2')
    norm2, _ = train_test_split(dataset3, test_size= 0.05, shuffle=False)


    logger.info('starting train_test_split concat')
    dataset = ConcatDataset([dataset1, norm1, norm2])



    logger.info('starting train_test_split final')
    train_ds, val_ds = train_test_split(dataset, test_size=0.1, random_state=2000)

    Datasets = {
        'train': train_ds,
        'test' : datasets.ImageFolder(test_folder, data_transforms['test']),
        'val'  : val_ds
    }

    Dataloaders = {
        'train': DataLoader(Datasets['train'], batch_size = 512, num_workers = 4),
        'test': DataLoader(Datasets['test'], batch_size = 512, shuffle = True, num_workers = 4),
        'val': DataLoader(Datasets['val'], batch_size = 512, shuffle = True, num_workers = 4),
    }

    files = []
    categories = []
    filenames = os.listdir(os.path.join(train_folder,'NORMAL'))
    for name in filenames:
        files.append(os.path.join(train_folder, 'NORMAL', name))
        categories.append('NORMAL')

    filenames = os.listdir(os.path.join(train_folder,'PNEUMONIA'))
    for name in filenames:
        files.append(os.path.join(train_folder, 'PNEUMONIA', name))
        categories.append('PNEUMONIA')

    Tr_PNEUMONIA = len([label for _, label in Datasets['train'] if label == 1])
    Tr_NORMAL = len(Datasets['train']) - Tr_PNEUMONIA
    V_PNEUMONIA = len([label for _, label in Datasets['val'] if label == 1])
    V_NORMAL = len(Datasets['val']) - V_PNEUMONIA
    Te_PNEUMONIA = len([label for _, label in Datasets['test'] if label == 1])
    Te_NORMAL = len(Datasets['test']) - Te_PNEUMONIA
    Pn = [Tr_PNEUMONIA, V_PNEUMONIA, Te_PNEUMONIA]
    No = [Tr_NORMAL, V_NORMAL, Te_NORMAL]


    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    model= nn.DataParallel(model)
    model.to(device)

    def trainer_loop(model, 
                    trainloader, 
                    loss_fn, 
                    optimizer, 
                    scheduler = None, 
                    t_gpu = True):
        model.train()
        tr_loss, tr_acc = 0.0, 0.0
        for i, data in enumerate(tqdm(trainloader)):
            img, label = data
            if t_gpu:
                    img, label = img.cuda(), label.cuda()
            optimizer.zero_grad()
            output = model(img)
            _, pred = torch.max(output.data, 1)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            
            tr_loss += loss.item()
            tr_acc += torch.sum(pred == label.data)
            torch.cuda.empty_cache()

        scheduler.step() if scheduler != None else None
        return tr_loss/len(trainloader.dataset), 100*tr_acc/len(trainloader.dataset)

    def val_loop(model, val_loader, loss_fn, t_gpu=True):
        model.train(False)
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                img, label = data
                if t_gpu:
                        img, label = img.cuda(), label.cuda()
                output = model(img)
                _, pred = torch.max(output.data, 1)
                loss = loss_fn(output, label)

                val_loss += loss.item()
                val_acc += torch.sum(pred == label.data)

        return val_loss/len(val_loader.dataset), 100*val_acc/len(val_loader.dataset)

    def train_model(epochs, 
                    model, 
                    trainloader, 
                    valloader, 
                    loss_fn, 
                    optimizer, 
                    scheduler = None, 
                    t_gpu = True):
        stat_dict = {
            'learning_rate':[],
            'train_loss':    [],
            'train_acc':     [],
            'val_loss':      [],
            'val_acc':       []    
        }
        print('*'*5+'Training Started'+'*'*5)
        for ep in range(epochs):
            print(f'Training epoch: {ep+1}')
            t_loss, t_acc = trainer_loop(
                model, trainloader, loss_fn, optimizer, scheduler, t_gpu
            )
            v_loss, v_acc = val_loop(
                model, valloader, loss_fn, t_gpu
            )
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
            print(f'Training   : Loss: {t_loss}    Accuracy: {t_acc}%')
            print(f'Validation : Loss: {v_loss}    Accuracy: {v_acc}%')
            stat_dict['learning_rate'].append(optimizer.param_groups[0]["lr"])
            stat_dict['train_loss'].append(t_loss)
            stat_dict['val_loss'].append(v_loss)
            stat_dict['train_acc'].append(t_acc)
            stat_dict['val_acc'].append(v_acc)
        print('Finished Training')
        return stat_dict

    epochs = 1
    alpha = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size = 3, 
                                                gamma = 0.1)
    loss_fn = nn.CrossEntropyLoss()


    
    logger.info(f'''starting image training for {epochs} epochs''')
    hist = train_model(epochs, 
                    model, 
                    Dataloaders['train'], 
                    Dataloaders['val'], 
                    loss_fn, 
                    optimizer,  
                    scheduler, 
                    device == 'cuda')


    @torch.no_grad()
    def test_loop(model, testdata, loss_fn, t_gpu):
        print('*'*5+'Testing Started'+'*'*5)
        model.train(False)
        model.eval()
        
        full_pred, full_lab = [], []
        
        TestLoss, TestAcc = 0.0, 0.0
        for data, target in testdata:
            if t_gpu:
                data, target = data.cuda(), target.cuda()
            model.to('cuda')
            output = model(data)
            loss = loss_fn(output, target)

            _, pred = torch.max(output.data, 1)
            TestLoss += loss.item() * data.size(0)
            TestAcc += torch.sum(pred == target.data)
            torch.cuda.empty_cache()
            full_pred += pred.tolist()
            full_lab += target.data.tolist()

        TestLoss = TestLoss / len(testdata.dataset)
        TestAcc = TestAcc / len(testdata.dataset)
        print(f'Loss: {TestLoss} Accuracy: {TestAcc}%')
        return full_pred, full_lab, output

    print('building testset to test model')
    logger.info('building testset to test model')
    testset = datasets.ImageFolder(test_folder, 
                            transform=transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),                                                              
                                                    transforms.ToTensor(),
                                                    ]))
    test_dl = DataLoader(testset, batch_size=32)


    logger.info('starting test loop.... almost done')
    pred, lab, output = test_loop(model, test_dl, loss_fn, True)


    # Save the trained model
    torch.save(model.state_dict(), 'pneumonia_detection.pt')

    from snowflake.ml.registry import Registry
    from snowflake.ml._internal.utils import identifier
    

    db = identifier._get_unescaped_name(session.get_current_database())
    schema = identifier._get_unescaped_name(session.get_current_schema())



    native_registry = Registry(session=session, database_name=db, schema_name=schema)

    
    logger.info('saving the model to the repo')

    try:
        # This can throw an exception if the model doesn't exist
        native_registry.delete_model(model_name)
        #native_registry.delete_model(model_name=model_name,model_version=snowflake_model_version)
    except:
        None
    snow_model = native_registry.log_model(
        model=model.module.to('cpu'),
        model_name=model_name,
        version_name=snowflake_model_version,
        sample_input_data=[val_ds[0][0].unsqueeze(0)],
        conda_dependencies=["absl-py",
                            "anyio",
                            "cloudpickle",
                            "numpy",
                            "packaging",
                            "pandas",
                            "pyyaml",
                            "snowflake-snowpark-python",
                            "typing-extensions",
                            "pytorch",
                            "torchvision"],
        options={"embed_local_ml_library": True, # This option is enabled to pull latest dev code changes.
                "relax": True}, 
    )
       
    

    logger.info('Training completed and model logged to registry')                    

       
except Exception as e:
    logger.error(f"An error was encountered: {e}", exc_info=True)
    
finally:
    session.close()
