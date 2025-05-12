import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    #storing the filepath for the preprocessor file
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ['Year_of_manufacture', 'Mileage', 'Engine_size']
            categorical_columns = [
                'fuel_type', 'gear_type', 'Make', 'Condition','Selling_Condition','Bought_Condition'
            ]
            #pipeline for numerical columns
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )
            #pipeline for categorical columns

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            #put all the pipelines into a column transformer
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]

            )
            #return preprocessor object to save for later use
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            #build preprocessing object
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="price"
            numerical_columns = ['Year_of_manufacture', 'Mileage', 'Engine_size']

            #split the training and testing data into their respective features and target columns
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # use preprocessor object to fit and transform the features
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df) 

            # it was giving a sparse matrix so had to change here
            input_feature_train_arr = input_feature_train_arr.toarray()
            input_feature_test_arr = input_feature_test_arr.toarray()

            # concatenate the features and their target variables
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object.")

            #save the preprocessing object
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            #return the training and the testing array, also the path to the preprocessing object
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)