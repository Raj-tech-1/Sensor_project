from flask import Flask, render_template, request, send_file
import os
import sys
from werkzeug.utils import secure_filename
from src.exception import CustomException
from src.logger import logging as lg
from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def home():
    return "Welcome to Wafer-predictor"

@app.route("/train")
def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return "Training Completed."
    except Exception as e:
        raise CustomException(e, sys)

@app.route('/predict', methods=['POST', 'GET'])
def upload():
    try:
        if request.method == 'POST':
            uploaded_file = request.files.get('file')

            if not uploaded_file or uploaded_file.filename == '':
                return "No file selected for uploading", 400

            prediction_pipeline = PredictionPipeline(uploaded_file)  # pass just the file
            prediction_file_detail = prediction_pipeline.run_pipeline()

            lg.info("Prediction completed. Downloading prediction file.")

            return send_file(
                prediction_file_detail.prediction_file_path,
                download_name=prediction_file_detail.prediction_file_name,
                as_attachment=True
            )
        else:
            return render_template('upload_file.html')
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
