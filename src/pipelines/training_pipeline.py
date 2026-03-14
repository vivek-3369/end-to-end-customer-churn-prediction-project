from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

import sys 
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

class TrainingPipeline:

    def __init__(self) :
        pass

    def start_training_pipeline(self):

        try:

            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

            model_trainer = ModelTrainer()
            model_report, best_model = model_trainer.initiate_model_trainer(train_arr, test_arr)

            model_evaluation = ModelEvaluation()
            metrics = model_evaluation.initiate_model_evaluation(test_arr)

            print("Model Report:", model_report)
            print("Best Model:", best_model)
            print("Model Evaluation Metrics")
            for i in metrics :
                print(f"{i}: {metrics[i]}")

            logging.info("Training Pipeline Completed")

        except Exception as e:
            raise CustomException(e, sys)