import shutil
import os
import sys
import pandas as pd
import pickle
from src.logger import logging
from src.exception import CustomException
from src.constant import TARGET_COLUMN, artifact_folder
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    prediction_output_dirname: str = "predictions"
    prediction_file_name: str = "prediction_file.csv"
    model_file_path: str = os.path.join(artifact_folder, 'model.pkl')
    preprocessor_path: str = os.path.join(artifact_folder, 'preprocessor.pkl')
    prediction_file_path: str = os.path.join(prediction_output_dirname, prediction_file_name)

class PredictionPipeline:
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
        self.prediction_pipeline_config = PredictionPipelineConfig()
        self.utils = MainUtils()  # Assuming MainUtils has your load_object method

    def save_input_files(self):
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, self.uploaded_file.filename)
        self.uploaded_file.save(file_path)
        return file_path

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Loading model and preprocessor...")
            model = self.utils.load_object(self.prediction_pipeline_config.model_file_path)
            preprocessor = self.utils.load_object(self.prediction_pipeline_config.preprocessor_path)

            logging.info("Transforming input features...")
            transform_x = preprocessor.transform(features)

            logging.info("Generating predictions...")
            preds = model.predict(transform_x)

            return preds
        except Exception as e:
            logging.error("Prediction failed", exc_info=True)
            raise CustomException(e, sys)

    def get_predicted_dataframe(self, input_dataframe_path: str):
        try:
            prediction_column_name = TARGET_COLUMN

            logging.info(f"Reading input CSV from {input_dataframe_path}...")
            input_dataframe = pd.read_csv(input_dataframe_path)

            # Drop 'Unnamed: 0' if present
            if "Unnamed: 0" in input_dataframe.columns:
                input_dataframe = input_dataframe.drop(columns="Unnamed: 0")

            predictions = self.predict(input_dataframe)

            input_dataframe[prediction_column_name] = [pred for pred in predictions]

            # Map predictions to labels
            target_column_mapping = {0: 'Faulty', 1: 'Good'}
            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)

            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname, exist_ok=True)

            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index=False)

            logging.info(f"Predictions completed and saved at {self.prediction_pipeline_config.prediction_file_path}")
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)
            return self.prediction_pipeline_config
        except Exception as e:
            raise CustomException(e, sys)
