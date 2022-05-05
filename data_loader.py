import pandas as pd 
import numpy as np
from pip import main
from scipy.stats import norm
import sqlite3 as sq3
import pandas.io.sql as pds
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy import stats
import dataframe_image as dfi 

'''
This class does the following:
- Loads tsv datasets
- Loads database as dataframe
- Checks for missing data 
- Drops rows with missing fields 
- Checks for datatypes 
- Formats timestamps as datetime objects, formats numbers as float 
'''


class Dataprocess:

    def __init__(self, meal_path, cgm_path, db_path, clinical_metricslist) -> None:
        self.meal_path= meal_path
        self.cgm_path= cgm_path
        self.db_path= db_path
        self.clinical_metrics= clinical_metricslist

    def db_data(self) ->None:
        con = sq3.Connection(self.db_path)
        query = """
                SELECT *
                FROM clinical;
                """
        self.clinical_df = pd.read_sql(query, con).reset_index()
        self.clinical_df= self.clinical_df.replace(['', 'NA'], [np.nan, np.nan])

    def read_tsv(self) -> None:
        self.meal_df= pd.read_csv(self.meal_path, sep='\t')
        self.cgm_df= pd.read_csv(self.cgm_path, sep='\t')

    def check_missing(self, df, savename)-> pd.DataFrame:
        total =df.isnull().sum().sort_values(ascending=False)
        percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        dfi.export(missing_data, savename)
        return missing_data

    def drop_missing(self) -> None:
        self.meal_df_clean= self.meal_df.dropna()
        self.cgm_df_clean= self.cgm_df.dropna()

    
    def drop_missing_clinical(self) -> pd.DataFrame:
        self.clinical_clean= (self.clinical_df.drop(['SSPG', 'Insulin_rate_dd','Height', 'Weight'], axis=1)).dropna()
        return self.clinical_clean


    def format_cgmdata(self) -> pd.DataFrame:
        self.cgm_df_clean['disp_time'] = pd.to_datetime(self.cgm_df_clean['DisplayTime'], format='%Y-%m-%d %H:%M:%S')
        self.cgm_formatted= self.cgm_df_clean.drop(['DisplayTime', 'InternalTime'], axis=1)
        self.cgm_formatted['GlucoseValue']= pd.to_numeric(self.cgm_df_clean['GlucoseValue'], errors='coerce')
        return self.cgm_formatted


    def format_mealdata(self) -> pd.DataFrame:
        self.meal_df_clean['time'] = pd.to_datetime(self.meal_df_clean['time'], format='%Y-%m-%d %H:%M:%S')
        self.meal_df_clean['GlucoseValue']= pd.to_numeric(self.meal_df_clean['GlucoseValue'], errors='coerce')
        self.meal_formatted= self.meal_df_clean
        return self.meal_formatted
        
    
