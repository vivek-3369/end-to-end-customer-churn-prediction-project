import os
import sys
import json
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

@dataclass
class ModelEvaluationConfig:
    metrics_file_path: str = os.path.join("artifacts", "metrics.json")
    model_path: str = os.path.join("artifacts", "model.pkl")

class ModelEvaluation:
    def __init__(self):
        self.config = ModelEvaluationConfig()

    def initiate_model_evaluation(self, test_arr):
        try:
            logging.info("Model Evaluation Started")
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Load the trained model from artifacts
            model = load_obj(self.config.model_path)

            logging.info("Predicting on test data")
            prediction = model.predict(X_test)
            prediction_prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, prediction)
            mat = confusion_matrix(y_test, prediction).tolist() # converting array to list for JSON serialization
            precision = precision_score(y_test, prediction)
            recall = recall_score(y_test, prediction)
            f1 = f1_score(y_test, prediction)
            roc_auc = roc_auc_score(y_test, prediction_prob)

            metrics = {
                "Accuracy": round(acc, 2),
                "Precision": round(precision, 2),
                "Recall": round(recall, 2),
                "F1_Score": round(f1, 2),
                "ROC-AUC Score": round(roc_auc, 2),
                "Confusion Matrix": mat,
            }

            
            with open(self.config.metrics_file_path, "w") as f:
                json.dump(metrics, f, indent=4)
            logging.info(f"Metrics saved to {self.config.metrics_file_path}")
            
            logging.info("Model Evaluation Completed")

            return metrics
            
        except Exception as e:
            raise CustomException(e, sys)