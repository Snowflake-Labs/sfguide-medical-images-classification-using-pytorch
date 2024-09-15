## ------------------------------------------------------------------------------------------------
# Copyright (c) 2023 Snowflake Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.You may obtain 
# a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0
    
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions andlimitations 
# under the License.
## ------------------------------------------------------------------------------------------------

'''
    Holds common utility functions, used across multiple scripts.
'''
import logging ,os ,json ,sys ,configparser
from pathlib import Path
import datetime
import pandas as pd
from snowflake.snowpark.session import Session
import snowflake.snowpark.functions as F

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p',
                    level=logging.INFO)

PROJECT_HOME_DIR='.'

def get_token():
    with open('/snowflake/session/token', 'r') as f:
        return f.read()

def get_connection_info(p_oauth_token):
    base_connection_params = {
        'account': os.getenv('SNOWFLAKE_ACCOUNT'),
        'host': os.getenv('SNOWFLAKE_HOST'),
        # 'protocol': os.getenv('SNOWFLAKE_PROTOCOL'),
        # 'port': os.getenv('SNOWFLAKE_PORT'),
        'database': os.getenv('SNOWFLAKE_DATABASE'),
        'schema': os.getenv('SNOWFLAKE_SCHEMA'),
        'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE')
        ,'role': os.getenv('SNOWFLAKE_ROLE')
        ,'authenticator': 'oauth'
        ,'token': p_oauth_token
    }
    return base_connection_params

def connect_to_snowflake() -> Session:
    logger.info('Connecting to snowflake ...')
    oauth_token = get_token()
    conn_info = get_connection_info(oauth_token)
    sp_session = Session.builder.configs(conn_info).create()

    return sp_session


def get_snowpark_dataframe(p_session: Session ,p_df: pd.DataFrame):
    # Convert the data frame into Snowpark dataframe, needed for merge operation
    sp_df = p_session.createDataFrame(p_df)

    # The column names gets defined in the snowpark dataframe in a case sensitive manner
    # hence rename them into a non case sensitive manner
    for c in p_df.columns:
        sp_df = sp_df.with_column_renamed(F.col(f'"{c}"'), c.upper())

    return sp_df

def get_config(p_project_home_dir: str)  -> configparser.ConfigParser:
    config_fl = f'{p_project_home_dir}/config.ini'
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    
    with open(config_fl) as f:
        config.read(config_fl)

    return config