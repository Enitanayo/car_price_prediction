import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
# from src.logger import lj



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        fuel_type: str,
        gear_type: str,
        Make:str,
        Year_of_manufacture: int,
        Condition: str,
        Mileage: int,
        Engine_size: int,
        Selling_Condition: str,
        Bought_Condition: str):

        self.fuel_type = fuel_type

        self.gear_type = gear_type

        self.Make = Make

        self.Year_of_manufacture = Year_of_manufacture

        self.Condition = Condition

        self.Mileage = Mileage

        self.Engine_size = Engine_size
        
        self.Selling_Condition = Selling_Condition
        
        self.Bought_Condition = Bought_Condition

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "fuel_type": [self.fuel_type],
                "gear_type": [self.gear_type],
                "Make": [self.Make],
                "Year_of_manufacture": [self.Year_of_manufacture],
                "Condition": [self.Condition],
                "Mileage": [self.Mileage],
                "Engine_size": [self.Engine_size],
                'Selling_Condition': [self.Selling_Condition],
                'Bought_Condition': [self.Bought_Condition],
            }

            # lj.info(pd.DataFrame(custom_data_input_dict))
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)