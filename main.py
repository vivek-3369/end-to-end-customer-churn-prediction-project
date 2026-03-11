from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.components.data_transformation import DataTransformation,DataTransformationConfig

import sys 
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__" :
    
    logging.info("The execution has started")

    try:
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()


        data_transofrmation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    except Exception as e:
        raise CustomException(e, sys)