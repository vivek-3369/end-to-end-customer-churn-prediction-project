from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion,DataIngestionConfig

import sys 
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__" :
    
    logging.info("The execution has started")

    try:
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()


    except Exception as e:
        raise CustomException(e, sys)