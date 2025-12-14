import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Encode target variable
            y_train = np.where(y_train == 'Yes', 1, 0)
            y_test = np.where(y_test == 'Yes', 1, 0)

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                ),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "XGBoost": XGBClassifier(eval_metric="logloss"),
                "CatBoost": CatBoostClassifier(verbose=False),
                "AdaBoost": AdaBoostClassifier()
            }

            params = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["liblinear"]
                },
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10]
                },
                "Decision Tree": {
                    "max_depth": [None, 10]
                },
                "XGBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100]
                },
                "CatBoost": {
                    "depth": [6, 8],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [50, 100]
                },
                "AdaBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100]
                }
            }

            # Evaluate models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                f"Best Model: {best_model_name} | ROC-AUC: {best_model_score}"
            )

            # RETRAIN BEST MODEL ON FULL DATA
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            # Save trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Final evaluation
            y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_test_pred_proba)

            return roc_auc

        except Exception as e:
            raise CustomException(e, sys)
