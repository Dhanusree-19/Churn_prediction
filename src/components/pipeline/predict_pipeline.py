import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictionPipeline:
    def __init__(self):
        self.model = load_object("artifacts/model.pkl")
        self.preprocessor = load_object("artifacts/preprocessor.pkl")

    def predict(self, data: pd.DataFrame):
        """
        Returns class label: Churn / No Churn
        """
        try:
            transformed_data = self.preprocessor.transform(data)
            prediction = self.model.predict(transformed_data)
            return "Churn" if prediction[0] == 1 else "No Churn"

        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, data: pd.DataFrame):
        """
        Returns churn probability
        """
        try:
            transformed_data = self.preprocessor.transform(data)

            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(transformed_data)
                return proba
            else:
                raise CustomException("Model does not support predict_proba()", sys)

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        SeniorCitizen: int,
        Partner: str,
        Dependents: str,
        tenure: int,
        PhoneService: str,
        MultipleLines: str,
        InternetService: str,
        OnlineSecurity: str,
        OnlineBackup: str,
        DeviceProtection: str,
        TechSupport: str,
        StreamingTV: str,
        StreamingMovies: str,
        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str,
        MonthlyCharges: float,
        TotalCharges: float
    ):
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame({
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
            })

        except Exception as e:
            raise CustomException(e, sys)


# Optional alias
PredictPipeline = PredictionPipeline
