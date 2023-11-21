# from data_ingestion import DataIngestion

# train   DataIngestion.initiate_data_ingestion()
import sys

import numpy as np
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from utils import save_object

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
import os

@dataclass
class DataTransformationConfig:
    prerpcessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    """
    The below function is responsible for the data transformation
    """

    def get_data_transformer_object(self):
        try:
            #selected numerical features
            numerical_features = ['reading score', 'writing score']

            #selected categorical features
            categorical_features = ['gender', 
                                    'race/ethnicity', 
                                    'parental level of education', 
                                    'lunch',
                                    'test preparation course']
            
            #created numerical features pipeline
            numerical_feature_pipeline = Pipeline(
                steps = [
                    ("impyter", SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            #created categorical features pipeline
            categorical_feature_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy = "most_frequent")),
                    ('onehotencoder', OneHotEncoder()),
                    ('stdscaler', StandardScaler(with_mean=False))

                ]
            )

            #Created log for above steps
            logging.info('Numerical columns standard scaling completed')
            logging.info('Categorical columns encoding completed')
            
            # Created a preprocessor object using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_feature_pipeline, numerical_features),
                    ('cat_pipeline', categorical_feature_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    """
    The below funciton is responsible for the initiation of data transformation 
    """

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and test data is completed")
            logging.info('Obtaining preprocessing Object')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math score"

            #training data dependent and independent variables 
            input_feature_train_df = train_df.drop(columns= [target_column_name],axis= 1)
            target_feature_train_df = train_df[target_column_name]

            #Test data dependent and independent variables
            input_features_test_df = test_df.drop(columns= [target_column_name],axis= 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # fit_transform and transform returns the numpy array or sparse matrix
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)
            
            #concatenate on the second axis(columns)
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f'Saved preprocessing Object')
            
            #here we are saving the pickle file
            save_object(
                file_path = self.data_transformation_config.prerpcessor_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.prerpcessor_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
            

