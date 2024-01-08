import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

#very important class
from dataclasses import dataclass

#importing from data_transformation
from data_transformation import DataTransformationConfig
from data_transformation import DataTransformation

#Importing model_trainer 
# from model_trainner import ModelTrainerConfig
# from model_trainner import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','raw.csv')


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            #Reading the data from csv file
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
     
            logging.info('Read the dataset as the dataframe') #logging

            #creating the folder 'artifacts'
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            #creating raw data path
            df.to_csv(self.ingestion_config.raw_data_path, index= False, header= True)

            #Doing train test split 
            train_set, test_set = train_test_split(df, test_size=0.2, random_state= 42)

            #creating train and test data paths
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('Ingestion of the data is completed') #logging

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    # Data ingestiom
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Data transformations
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)
    print(train_arr)

    # Model trainner 
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))