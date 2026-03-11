import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file: str = os.path.join("artifacts", "preprocessor.pkl")

class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply feature engineering steps from the notebook.
    This ensures that train and test data go through the exact same transformations
    without data leakage.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # Nothing to fit for these custom transformations

    def transform(self, X):
        try:
            # We create a copy to avoid SettingWithCopyWarning
            X_transformed = X.copy()
            
            # Drop customerID
            if "customerID" in X_transformed.columns:
                X_transformed = X_transformed.drop("customerID", axis=1)

            # Convert TotalCharges to numeric
            if "TotalCharges" in X_transformed.columns:
                X_transformed["TotalCharges"] = pd.to_numeric(X_transformed["TotalCharges"], errors="coerce")
                
                # Impute missing TotalCharges with 0 or drop them. 
                # Since dropping rows in transform can be risky (in a pipeline), filling with 0 or mean is better.
                # In the notebook, you dropped NA. We will fill with 0 to keep the shape consistent for inference.
                X_transformed["TotalCharges"] = X_transformed["TotalCharges"].fillna(0)

            # Replace "No phone service" and "No internet service" with "No"
            columns_to_transform = [
                "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"
            ]
            for col in columns_to_transform:
                if col in X_transformed.columns:
                    X_transformed[col] = X_transformed[col].str.replace("No phone service", "No").str.replace("No internet service", "No")

            # 1. Total Services Count
            services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
            if all(col in X_transformed.columns for col in services):
                X_transformed["Total_Services_Count"] = X_transformed[services].apply(lambda x: np.where(x == "Yes", 1, 0)).apply(lambda row: (row > 0).sum(), axis=1)

            # 2. Historical Monthly Charges
            if "TotalCharges" in X_transformed.columns and "tenure" in X_transformed.columns:
                # Add a small epsilon to avoid division by zero
                X_transformed["Historical_Monthly_Charges"] = round(X_transformed["TotalCharges"] / (X_transformed["tenure"] + 1e-5), 2)
            else:
                X_transformed["Historical_Monthly_Charges"] = 0

            # 3. Charge Spike
            if "MonthlyCharges" in X_transformed.columns and "Historical_Monthly_Charges" in X_transformed.columns:
                X_transformed["Charge_Spike"] = X_transformed["MonthlyCharges"] - X_transformed["Historical_Monthly_Charges"]

            # 4. Total Security Tech Services
            security = ["OnlineSecurity", "OnlineBackup"]
            if all(col in X_transformed.columns for col in security):
                X_transformed["Total_Security_Tech_Services"] = X_transformed[security].apply(lambda x: np.where(x == "Yes", 1, 0)).apply(lambda row: (row > 0).sum(), axis=1)

            # 5. Total Streaming Services
            streaming = ["StreamingTV", "StreamingMovies"]
            if all(col in X_transformed.columns for col in streaming):
                X_transformed["TotalStreamingServices"] = X_transformed[streaming].apply(lambda x: np.where(x == "Yes", 1, 0)).apply(lambda row: (row > 0).sum(), axis=1)

            # 6. Is_Family
            if "Dependents" in X_transformed.columns and "Partner" in X_transformed.columns:
                X_transformed["Is_Family"] = np.where((X_transformed["Dependents"] == "Yes") | (X_transformed["Partner"] == "Yes"), 1, 0)

            # 7. Is_Autopay
            if "PaymentMethod" in X_transformed.columns:
                X_transformed["Is_Autopay"] = np.where(X_transformed["PaymentMethod"].str.contains("auto", case=False, na=False), 1, 0)

            # 8. Tenure Group
            def tenure_group(tenure):
                if tenure <= 6:
                    return "0-6 months (New)"
                elif 6 < tenure <= 24:
                    return "6-24 months"
                elif 24 < tenure <= 60:
                    return "24-60 months"
                else:
                    return "60+ months"
            
            if "tenure" in X_transformed.columns:
                X_transformed["Tenure_Group"] = X_transformed["tenure"].apply(tenure_group)

            # Drop redundant columns as per notebook
            cols_to_drop = ["gender", "PhoneService", "TotalCharges", "StreamingTV", "StreamingMovies", "OnlineSecurity", "OnlineBackup"]
            cols_to_drop_existing = [col for col in cols_to_drop if col in X_transformed.columns]
            X_transformed = X_transformed.drop(columns=cols_to_drop_existing)

            return X_transformed

        except Exception as e:
            raise CustomException(e, sys)


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self, X):
        """
        This function is responsible for building the numeric and categorical pipelines
        """
        try:
            # We determine categorical vs numerical/continuous AFTER custom feature engineering
            # Using the same logic as the notebook:
            numerical_features = [feature for feature in X.columns if X[feature].dtype != "O" and X[feature].dtype != "str"]
            categorical_features = [feature for feature in X.columns if X[feature].dtype == "O" or X[feature].dtype == "str"]
            
            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numerical_features}")

            # Note: Discrete vs Continuous separation from notebook is merged here 
            # into a single StandardScaler for numerical for simplicity, 
            # but you can separate them out if desired.

            numerical_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=False)) # with_mean=False in case of sparse matrices
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
                ]
            )

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numerical_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, numerical_features),
                    ("cat_pipeline", categorical_pipeline, categorical_features)
                ], remainder="passthrough"
            )

            # The final pipeline bundles custom feature engineering WITH the preprocessor 
            # so that both are executed automatically and sequentially
            final_pipeline = Pipeline([
                ("custom_feature_engineer", CustomFeatureEngineer()),
                ("preprocessor", preprocessor)
            ])

            return final_pipeline
        
        except Exception as e:
            raise CustomException(e, sys)
