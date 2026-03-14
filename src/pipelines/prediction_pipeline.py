import sys 
import pandas as pd 

from src.exception import CustomException
from src.utils import load_obj


class CustomData :
    
    def __init__(self, gender:str, seniorcitizen:str, partner:str, dependents:str, tenure:int, phoneservice:str, 
                multiplelines:str, internetservice:str, onlinesecurity:str, onlinebackup:str, deviceprotection:str, 
                techsupport:str, streamingtv:str, streamingmovies:str, contract:str, paperlessbilling:str, paymentmethod:str, 
                monthlycharges:float, totalcharges:float) :

        self.gender = gender
        self.seniorcitizen = seniorcitizen
        self.partner = partner
        self.dependents = dependents
        self.tenure = tenure
        self.phoneservice = phoneservice
        self.multiplelines = multiplelines
        self.internetservice = internetservice
        self.onlinesecurity = onlinesecurity
        self.onlinebackup = onlinebackup
        self.deviceprotection = deviceprotection
        self.techsupport = techsupport
        self.streamingtv = streamingtv
        self.streamingmovies = streamingmovies
        self.contract = contract
        self.paperlessbilling = paperlessbilling
        self.paymentmethod = paymentmethod
        self.monthlycharges = monthlycharges
        self.totalcharges = totalcharges


    def get_input_as_df(self) :

        try :
            input_dict = {
                "gender" : [self.gender],
                "SeniorCitizen" : [self.seniorcitizen],
                "Partner" : [self.partner],
                "Dependents" : [self.dependents],
                "tenure" : [self.tenure],
                "PhoneService" : [self.phoneservice],
                "MultipleLines" : [self.multiplelines],
                "InternetService" : [self.internetservice],
                "OnlineSecurity" : [self.onlinesecurity],
                "OnlineBackup" : [self.onlinebackup],
                "DeviceProtection" : [self.deviceprotection],
                "TechSupport" : [self.techsupport],
                "StreamingTV" : [self.streamingtv],
                "StreamingMovies" : [self.streamingmovies],
                "Contract" : [self.contract],
                "PaperlessBilling" : [self.paperlessbilling],
                "PaymentMethod" : [self.paymentmethod],
                "MonthlyCharges" : [self.monthlycharges],
                "TotalCharges" : [self.totalcharges]
            }

            input_df = pd.DataFrame(input_dict)

            return input_df

        except Exception as e :
            raise CustomException(e, sys)
        

class PredictionPipeline :

    def __init__(self) :
        pass

    def prediction(self, features) :
        '''
        This function is used to preprocess the input from the user and predict the output
        '''
        try :

            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            # Loading the object
            model = load_obj(file_path=model_path)
            preprocessor = load_obj(file_path=preprocessor_path)

            # Data Preprocessing for the user data
            columns_to_transform = ["MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection",
                                    "TechSupport","StreamingTV","StreamingMovies"]
            for column in columns_to_transform:
                if column in features.columns:
                    features[column] = features[column].str.replace("No phone service", "No").str.replace("No internet service", "No")

            if "SeniorCitizen" in features.columns:
                features["SeniorCitizen"] = features["SeniorCitizen"].replace({1: "Yes", 0: "No", "1": "Yes", "0": "No"})
            
            # Feature Engineering for the user data
            bins = [0, 12, 24, 48, 72]
            labels = ["New (0-12m)", "Growing (13-24m)", "Established (25-48m)", "Loyal (49-72m)"]
            features["tenure_group"] = pd.cut(features["tenure"], bins=bins, labels=labels, include_lowest=True).astype(str)

            features["avg_monthly_charges"] = features.apply(lambda r: r["MonthlyCharges"] if r["tenure"] == 0 else r["TotalCharges"] / r["tenure"], axis=1)
            features["charge_delta"] = features["MonthlyCharges"] - features["avg_monthly_charges"]

            addon_services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
            features["num_services"] = features[addon_services].apply(lambda row: (row == "Yes").sum(), axis=1)

            features["contract_risk"] = features["Contract"].map({"Month-to-month": 3, "One year": 2, "Two year": 1})
            features["is_electronic_check"] = (features["PaymentMethod"] == "Electronic check").astype(int)
            features["is_auto_payment"] = features["PaymentMethod"].str.contains("automatic", case=False).astype(int)
            features["high_friction_billing"] = ((features["is_auto_payment"] == 0) & (features["PaperlessBilling"] == "Yes")).astype(int)
            features["vulnerability_score"] = ((features["SeniorCitizen"] == "Yes").astype(int) + (features["Partner"] == "No").astype(int) + (features["Dependents"] == "No").astype(int))

            # Apply the preprocessor scaler and encoders
            data_scaled = preprocessor.transform(features)
            output = model.predict_proba(data_scaled)[0][1]

            return output

        except Exception as e :
            raise CustomException(e, sys)