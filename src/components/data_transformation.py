import os 
import sys
import pandas as pd 
import numpy as np 

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE


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

            # Ordinal columns - where categories have order 
            ordinal_cols = ["Contract", "tenure_group"]
            contract_order = ["Month-to-month", "One year", "Two year"]
            tenure_order = ["New (0-12m)", "Growing (13-24m)", "Established (25-48m)", "Loyal (49-72m)"]

            # nominal cols - where categories have no order
            nominal_cols = ["InternetService", "PaymentMethod"]

            # continuous numerical features to scale
            numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_charges', 'charge_delta']

            # engineered integer features — also need scaling for distance-based models
            engineered_cols = ['num_services', 'contract_risk', 'is_electronic_check',
                               'is_auto_payment', 'high_friction_billing', 'vulnerability_score']

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # same pipeline reused for engineered features
            engineered_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            ordinal_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(categories=[contract_order, tenure_order], handle_unknown="use_encoded_value", unknown_value=-1))
            ])

            nominal_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder())
            ])

            binary_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
                           "MultipleLines","OnlineSecurity", "OnlineBackup", "DeviceProtection",
                           "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]

            gender_category = ["Male", "Female"]
            yes_no_category = ["No", "Yes"]
            binary_categories = [gender_category] + [yes_no_category] * (len(binary_cols) - 1)

            binary_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(categories=binary_categories, handle_unknown="use_encoded_value", unknown_value=-1))
            ])

            logging.info(f"Ordinal Columns: {ordinal_cols}")
            logging.info(f"Nominal Columns: {nominal_cols}")
            logging.info(f"Numerical Columns: {numeric_cols}")
            logging.info(f"Engineered Columns: {engineered_cols}")
            logging.info(f"Binary Columns: {binary_cols}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numeric_cols),
                    ("engineered_pipeline", engineered_pipeline, engineered_cols),
                    ("ordinal_pipeline", ordinal_pipeline, ordinal_cols),
                    ("nominal_pipeline", nominal_pipeline, nominal_cols),
                    ("binary_pipeline", binary_pipeline, binary_cols)
                ],
                remainder="drop"  
            )

            return preprocessor

        except Exception as e :
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path) :

        try: 
            logging.info("Data Transformation Started")
            
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")


            logging.info("Data Preprocesing Started")
                        # Drop customerID
            if "customerID" in train_df.columns and "customerID" in test_df.columns:
                train_df = train_df.drop("customerID", axis=1)
                test_df = test_df.drop("customerID", axis=1)

            if "TotalCharges" in train_df.columns and "TotalCharges" in test_df.columns:
                train_df["TotalCharges"] = pd.to_numeric(train_df["TotalCharges"], errors="coerce")
                test_df["TotalCharges"] = pd.to_numeric(test_df["TotalCharges"], errors="coerce")

                train_df = train_df.dropna()
                test_df = test_df.dropna()

            columns_to_transform = ["MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection",
                                    "TechSupport","StreamingTV","StreamingMovies"]
            
            for column in columns_to_transform :
                if column in train_df.columns and column in test_df.columns  :
                    train_df[column] = train_df[column].str.replace("No phone service", "No").str.replace("No internet service", "No")
                    test_df[column] = test_df[column].str.replace("No phone service", "No").str.replace("No internet service", "No")

            if "tenure_group" in train_df.columns and "tenure_group" in test_df.columns:
                train_df["tenure_group"] = train_df["tenure_group"].astype(str)
                test_df["tenure_group"] = test_df["tenure_group"].astype(str)
                    
            if "SeniorCitizen" in train_df.columns and "SeniorCitizen" in test_df.columns :
                train_df["SeniorCitizen"] = train_df["SeniorCitizen"].map({1: "Yes", 0: "No"})
                test_df["SeniorCitizen"] = test_df["SeniorCitizen"].map({1: "Yes", 0: "No"})

            logging.info("Data Preprocessing Completed")


            logging.info("Feature Engineering Started")
            logging.info("Feature Creation")

            # 1. Tenure Binning
            bins = [0, 12, 24, 48, 72]
            labels = ["New (0-12m)", "Growing (13-24m)", "Established (25-48m)", "Loyal (49-72m)"]
            train_df["tenure_group"] = pd.cut(train_df["tenure"], bins=bins, labels=labels, include_lowest=True)
            test_df["tenure_group"] = pd.cut(test_df["tenure"], bins=bins, labels=labels, include_lowest=True)
            
            # 2. average_monthly_charges = total_charges / tenure
            train_df["avg_monthly_charges"] = train_df.apply(lambda r: r["MonthlyCharges"] if r["tenure"] == 0 else r["TotalCharges"] / r["tenure"], axis=1)
            test_df["avg_monthly_charges"] = test_df.apply(lambda r: r["MonthlyCharges"] if r["tenure"] == 0 else r["TotalCharges"] / r["tenure"], axis=1)


            # 3. charge_delta : how much current bill deviates from the lifetime average 
            train_df["charge_delta"] = train_df["MonthlyCharges"] - train_df["avg_monthly_charges"]
            test_df["charge_delta"] = test_df["MonthlyCharges"] - test_df["avg_monthly_charges"]

            # 4. Service Count
            addon_services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                  "TechSupport", "StreamingTV", "StreamingMovies"]
            train_df["num_services"] = train_df[addon_services].apply(lambda row: (row == "Yes").sum(), axis=1)
            test_df["num_services"] = test_df[addon_services].apply(lambda row: (row == "Yes").sum(), axis=1)

            # 5. Contract Risk Score
            train_df["contract_risk"] = train_df["Contract"].map({"Month-to-month": 3, "One year": 2, "Two year": 1})
            test_df["contract_risk"] = test_df["Contract"].map({"Month-to-month": 3, "One year": 2, "Two year": 1})

            # 6. is_electronic_payment
            train_df["is_electronic_check"] = (train_df["PaymentMethod"] == "Electronic check").astype(int)
            test_df["is_electronic_check"] = (test_df["PaymentMethod"] == "Electronic check").astype(int)

            # 7. is_auto_payment
            train_df["is_auto_payment"] = train_df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
            test_df["is_auto_payment"] = test_df["PaymentMethod"].str.contains("automatic", case=False).astype(int)

            # 8.  high friction billing = non autopayment and paperless have higher risk of churn
            train_df["high_friction_billing"] = ((train_df["is_auto_payment"] == 0) & (train_df["PaperlessBilling"] == "Yes")).astype(int)
            test_df["high_friction_billing"] = ((test_df["is_auto_payment"] == 0) & (test_df["PaperlessBilling"] == "Yes")).astype(int)


            # 9. Vulnerability Score - SeniorCitizen with no Partners and no Dependents have higher risk of churn
            train_df["vulnerability_score"] = ((train_df["SeniorCitizen"] == "Yes").astype(int) +(train_df["Partner"] == "No").astype(int) +(train_df["Dependents"] == "No").astype(int))
            test_df["vulnerability_score"] = ((test_df["SeniorCitizen"] == "Yes").astype(int) +(test_df["Partner"] == "No").astype(int) +(test_df["Dependents"] == "No").astype(int))
            logging.info("Feature Creation Completed")

            logging.info("Splitting the input and output features")
            target_feature = "Churn"
            input_features_train_df = train_df.drop(target_feature, axis=1)
            target_feature_train_df = train_df[target_feature]

            input_features_test_df = test_df.drop(target_feature, axis=1)
            target_feature_test_df = test_df[target_feature]

            logging.info("Feature Encoding")

            preprocessing_obj = self.get_data_transformer_obj()

            logging.info("Applying preprocessing on training and testing data")
            logging.info("Feature Encoding")

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            target_feature_train_arr = target_feature_train_df.map({"No": 0, "Yes": 1})
            target_feature_test_arr = target_feature_test_df.map({"No": 0, "Yes": 1})

            logging.info("Applying SMOTE to the training dataset")
            smote = SMOTE(sampling_strategy="minority", random_state=42)
            input_features_train_resampled, target_feature_train_resampled = smote.fit_resample(input_features_train_arr, target_feature_train_arr)
            
            # Combining the transformed input features with target feature
            train_arr = np.c_[
                input_features_train_resampled, np.array(target_feature_train_resampled)
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