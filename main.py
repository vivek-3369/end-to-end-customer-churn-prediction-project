import sys 
from src.logger import logging
from src.exception import CustomException

from src.pipelines.training_pipeline import TrainingPipeline


if __name__ == "__main__" :

    try :
        logging.info("Training Pipeline Started")

        training_pipeline = TrainingPipeline()
        training_pipeline.start_training_pipeline()

    except Exception as e:
        raise CustomException(e, sys)