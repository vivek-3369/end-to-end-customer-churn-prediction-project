from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

import os 
import sys
import pandas as pd


@dataclass  # is a special class used mostly for holding data without writing boilerplate code and it automaticall generates __init__, __rep__, __eq__ 
class DataIngestionConfig:

    train_path:str = os.path.join("artifacts","train.csv")
    test_path:str = os.path.join("artifacts","test.csv")
    raw_path:str = os.path.join("artifacts","raw.csv")



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):

        try :
            df = pd.read_csv(os.path.join("notebooks/data","churn-data.csv"))
            logging.info("Reading data from a CSV File")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_path), exist_ok=True)  # Creates artifacts folder in the directory

            df.to_csv(self.ingestion_config.raw_path, header=True, index=False)

            train_data, test_data = train_test_split(df, test_size=0.2, random_state=1)
            logging.info("Splitted the dataset into train and test")

            train_data.to_csv(self.ingestion_config.train_path, header=True, index=False)
            test_data.to_csv(self.ingestion_config.test_path, header=True, index=False)

            logging.info("Saving the train and test data")
            logging.info("Data Ingestion Completed")


            return (
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
                    )

        except Exception as e :
            raise CustomException(e, sys)
