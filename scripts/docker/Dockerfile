# Start your image with a node base image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# The /app directory should act as the main application directory
WORKDIR /app

# Copy the app package and package-lock.json file
COPY common_utils.py /app
COPY config.ini /app
COPY modeltrainreg.py /app
RUN chmod 777 /app/modeltrainreg.py
 
# COPY config.ini /app
# COPY sflk_connection.json /app
# COPY modeltrainreg.py /app
# RUN chmod 777 /app/modeltrainreg.py

# Install node packages, install serve, build the app, and remove dependencies at the end
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg
RUN apt-get install -y python3 python3-pip
RUN pip3 install torch tqdm torchvision imblearn
RUN pip3 install mlxtend opencv-python onnx
RUN apt-get install libgl1 --yes 
RUN pip3 install snowflake-snowpark-python[pandas] snowflake-ml-python
RUN pip3 uninstall typing_extensions --yes
RUN pip3 install typing_extensions==4.7.1
RUN pip3 install importlib-metadata


ENTRYPOINT ["python3", "/app/modeltrainreg.py"]