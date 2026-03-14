import os 
import sys 

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_obj
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


@dataclass 
class ModelTrainerConfig :
    trained_model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer :

    def __init__(self) :
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr) :

        try:
            logging.info("Model Training Started")
            logging.info("Split training and testing data into input and output")

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:,-1],
                test_arr[:, :-1],
                test_arr[:,-1]
            )

            models = {
                    "Logistic Regression" : LogisticRegression(random_state=42),
                    "DecisionTreeClassifier" : DecisionTreeClassifier(random_state=42),
                    "KNeighborsClassifier" : KNeighborsClassifier(),
                    "Random Forest" : RandomForestClassifier(random_state=42),
                    "AdaBoost": AdaBoostClassifier(random_state=42),
                    "Gradient Boost": GradientBoostingClassifier(random_state=42)
            }

            model_report = evaluate_models(X_train, X_test, y_train, y_test, models)

            best_model_score = max(sorted(model_report.values()))
            best_model = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Raising an error if the best model accuracy is less than 0.6 
            if best_model_score < 0.6 :
                raise CustomException("No best model found")

            logging.info("Best model found on both training and test dataset")

            best_model_obj = models[best_model]

            logging.info("Saving Best Model Object")
            save_obj(
                self.model_trainer_config.trained_model_path,
                best_model_obj
            )
            logging.info("Model Training Completed")
            
            return model_report,best_model

        except Exception as e :
            raise CustomException(e, sys)