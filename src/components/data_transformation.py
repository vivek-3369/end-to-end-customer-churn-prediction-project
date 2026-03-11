from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj

import os 
import sys
import pandas as pd 
import numpy as np
from dataclasses import dataclass 

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file : str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:

    def __init__(self) :
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self) :
        '''
        This function is responsible for data transformation
        '''

        try :
            numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
            categorical_features = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
                                    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                                    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy = "median")),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy = "most_frequent")),
                ('onehot', OneHotEncoder(drop="first"))
            ])

            logging.info(f"Categorical Columns: {categorical_features}")
            logging.info(f"Numerical Columns: {numerical_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            , remainder="passthrough")

            return preprocessor

        except Exception as e :
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path) :

        try: 
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")


            logging.info("Data Preprocesing Started")
                        # Drop customerID
            if "customerID" in train_df.columns and "customerID" in test_df.columns:
                train_df = train_df.drop("customerID", axis=1)
                test_df = test_df.drop("customerID", axis=1)

            if "TotalCharges" in train_df.columns and "TotalCharges" in test_df:
                train_df["TotalCharges"] = pd.to_numeric(train_df["TotalCharges"], errors="coerce")
                test_df["TotalCharges"] = pd.to_numeric(test_df["TotalCharges"], errors="coerce")

                train_df = train_df.dropna()
                test_df = test_df.dropna()

            columns_to_transform = ["MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection",
                                    "TechSupport","StreamingTV","StreamingMovies"]
            
            for column in columns_to_transform :
                if column in train_df.columns and column in test_df :
                    train_df[column] = train_df[column].str.replace("No phone service", "No").str.replace("No internet service", "No")
                    test_df[column] = test_df[column].str.replace("No phone service", "No").str.replace("No internet service", "No")
                    
            logging.info("Data Preprocessing Completed")


            logging.info("Feature Engineering Started")
            
            logging.info("Splitting the input and output features")
            target_feature = "Churn"
            input_features_train_df = train_df.drop(target_feature, axis=1)
            target_feature_train_df = train_df[target_feature]

            input_features_test_df = test_df.drop(target_feature, axis=1)
            target_feature_test_df = test_df[target_feature]

            logging.info("Feature Encoding")

            preprocessing_obj = self.get_data_transformer_obj()


            logging.info("Applying preprocessing on training and testing data")
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            target_feature_train_arr = target_feature_train_df.map({"No": 0, "Yes": 1})
            target_feature_test_arr = target_feature_test_df.map({"No": 0, "Yes": 1})
            
            # Combining the transformed input features with target feature
            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_arr)
            ]

            test_arr = np.c_[
                input_features_test_arr, np.array(target_feature_test_arr)
            ]

            logging.info("Saving the preprocessor object")

            save_obj(self.data_transformation_config.preprocessor_obj_file, preprocessing_obj)

            logging.info("Data Transformation Completed")

            return (
                train_arr, 
                test_arr,   
                self.data_transformation_config.preprocessor_obj_file
            )

        except Exception as e :
            raise CustomException(e, sys)








