import os
import sys
import dill
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.exception import CustomException
from src.logger import logger

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logger.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logger.error("Error in saving object", exc_info=True)
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            report[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            logger.info(f"Model: {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        return report

    except Exception as e:
        logger.error("Error in evaluating models", exc_info=True)
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logger.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        logger.error("Error in loading object", exc_info=True)
        raise CustomException(e, sys)
