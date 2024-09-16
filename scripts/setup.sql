/***************************************************************************************************
 
Quickstart:   Medical Images Classification using PyTorch in Snowflake
Version:      v1
Author:       Kala Govindarajan
Copyright(c): 2024 Snowflake Inc. All rights reserved.
****************************************************************************************************
SUMMARY OF CHANGES
Date(yyyy-mm-dd)    Author              Comments
------------------- ------------------- ------------------------------------------------------------
2024-09-12          Kala Govindarajan      Initial Release
***************************************************************************************************/


--Setup
use role ACCOUNTADMIN;

create database dicom_DB;
create schema dicom_SCHEMA;
create warehouse dicom_WH_S WAREHOUSE_SIZE=SMALL;

use database dicom_DB;
use schema dicom_SCHEMA;
use warehouse dicom_WH_S;

create stage data_STAGE;
create stage model_STAGE;
create stage dicomapp_STAGE;
create stage service_STAGE;
create image repository dicom_REPO;



SHOW IMAGE REPOSITORIES like 'dicom_REPO' in database DICOM_DB;
create security integration if not exists SNOWSERVICES_INGRESS_OAUTH
  type=oauth
  oauth_client=snowservices_ingress
  enabled=true;

create compute pool DICOM_GPU3
min_nodes = 1
max_nodes = 2
instance_family = GPU_NV_S
auto_suspend_secs = 7200;

show compute pools;


grant usage on database dicom_DB to role SYSADMIN;
grant all on schema dicom_SCHEMA to role SYSADMIN;
grant create service on schema dicom_SCHEMA to role SYSADMIN;
grant usage on warehouse dicom_WH_S to role SYSADMIN;
grant READ,WRITE on stage data_STAGE to role SYSADMIN;
grant READ,WRITE on image repository dicom_REPO to role SYSADMIN;
grant all on compute pool dicom_GPU3 to role SYSADMIN;
grant bind service endpoint on account to role SYSADMIN;
grant monitor usage on account to role SYSADMIN;

CREATE OR REPLACE NETWORK RULE allow_all_rule
  TYPE = 'HOST_PORT'
  MODE= 'EGRESS'
  VALUE_LIST = ('0.0.0.0:443','0.0.0.0:80');

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION ALLOW_ALL_ACCESS_INTEGRATION
  ALLOWED_NETWORK_RULES = (allow_all_rule)
  ENABLED = true;

GRANT USAGE ON INTEGRATION ALLOW_ALL_ACCESS_INTEGRATION TO ROLE SYSADMIN;


show image repositories;

/*
---------Data Load---------
Download data from this S3 public bucket location or can directly use the one from the cloned Git repo setup/data folder:
s3://sfquickstarts/sfguide_medical_images_classification_using_pytorch/test/NORMAL/
s3://sfquickstarts/sfguide_medical_images_classification_using_pytorch/test/PNEUMONIA/
s3://sfquickstarts/sfguide_medical_images_classification_using_pytorch/train/NORMAL/
s3://sfquickstarts/sfguide_medical_images_classification_using_pytorch/train/PNEUMONIA/

*/
--Stop here--Build and Push the docker image to image repository, place the service.yaml in the stage and then proceed below--

--Execute Job
execute JOB service DICOM_GPU3
in compute pool 
EXTERNAL_ACCESS_INTEGRATIONS = (ALLOW_ALL_ACCESS_INTEGRATION)
NAME=DICOM_SERVICE_JOB
from @dicom_SCHEMA.service_stage specification_file ='service_definition.yaml';


-- Check Status
CALL SYSTEM$GET_SERVICE_STATUS('DICOM_SERVICE_JOB');
select * from TRAINING_LOG;
SELECT SYSTEM$GET_SERVICE_LOGS('DICOM_SERVICE_JOB', 0, 'pneumonia-rapid-service', 1000);
