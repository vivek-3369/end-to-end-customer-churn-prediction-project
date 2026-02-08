import os
from pathlib import Path # for creating paths
import logging 

logging.basicConfig(level=logging.INFO)

list_of_files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestions.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/pipelines/__init__.py",
    "src/pipelines/training_pipeline.py",
    "src/pipelines/prediction_pipeline.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
]

for file_path in list_of_files :
    file_path = Path(file_path)
    dir_name, file_name = os.path.split(file_path)

    if dir_name != "" :
        os.makedirs(dir_name, exist_ok=True)
        logging.info(f"Creating a Directory name {dir_name}")
    
    if not(os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as file_obj:
            pass
            logging.info(f"Creating an empty file: {file_path}")
    
    else :
        logging.info(f"File already exists")