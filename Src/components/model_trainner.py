import os
import sys
from dataclasses import dataclass

# Importing Machine learning Algorithms
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Importing exception, utils and logging
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import eval_model, save_object

from sklearn.metrics import r2_score


# MODEL TRAINING 

@dataclass
class ModelTrainerConfig:
    trainedModelFilePath = os.path.join('artifacts','modeltrainer.pkl')

# Here we do the model training
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig
    
    # The paramerters are the output from the Data transformation
    def initiate_model_trainer(self, training_arr, test_arr):
        try:
            logging.info('Splitting Training and test input data')

            #Splitting the train and test input data
            X_train, y_train, X_test, y_test = (
                training_arr[:, :-1], 
                training_arr[:, -1], 
                test_arr[:, :-1],
                test_arr[:,-1]
            )

            # Testing all the models
            # Dictionary of models
            models = {
                # 'Linear Regression': LinearRegression(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'Xgboost' : XGBRegressor(),
                'Catboost': CatBoostRegressor(),
                'Adaboost': AdaBoostRegressor()               
            }

            # Dictionary for hyperparameters
            params={
                "Decision Tree Regressor": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting Regressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Xgboost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Catboost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Adaboost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            # eval_model is the function in the utils module
            model_report: dict = eval_model(X_train = X_train, y_train = y_train, X_test= X_test, y_test= y_test, models = models, params = params)

            ## Get the best model score from the dictionary 
            best_model_score =  max(sorted(model_report.values()))

            # getting the best model name from the dictionary 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            # Dumping my best model
            # Pickle file is created till here
            save_object(
                file_path = self.model_trainer_config.trainedModelFilePath,
                obj = best_model
            )

            # Now we do the predictions

            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test, predicted)

            return r2

        except Exception as e:
            raise CustomException(e,sys)
            
    