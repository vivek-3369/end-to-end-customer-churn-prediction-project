import os 
import sys 

from src.exception import CustomException
from src.logger import logging

import pickle
from sklearn.metrics import roc_auc_score



def save_obj(file_path, obj) :

    try :
        logging.info("Saving object")
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e :
        raise CustomException(e, sys)


def evaluate_models(x_train, x_test, y_train, y_test, models) :

    try:
        report = {}

        for i in range(len(models)) :
            
            model = list(models.values())[i]
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)

            roc_score = roc_auc_score(y_test, y_pred)

            report[list(models.keys())[i]] = roc_score

        return report

    except Exception as e :
        raise CustomException(e, sys)


def load_obj(file_path) :

    try:
        with open(file_path, "rb") as file_obj :
            return pickle.load(file_obj)
    except Exception as e :
        raise CustomException(e, sys)