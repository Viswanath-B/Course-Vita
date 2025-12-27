# ECG Signal Classification API

This project implements a Flask-based REST API to deploy a trained machine learning model for ECG signal classification.  
The API accepts numerical ECG feature inputs and returns the predicted heart condition class.

## Project Overview

- **Model Type:** ECG Signal Classification  
- **Framework:** Flask  
- **Language:** Python  
- **Input:** Numerical ECG features  
- **Output:** ECG class label (e.g., NSR, ARR, AFF, CHF)

## Project Structure

# Files:
main.py
ecg_model.pkl
requirements.txt

# Steps:
pip install -r requirements.txt
python app.py

# API:
POST /predict

# Prediction:

- The deployed ECG model successfully generates ECG class predictions.
- Prediction outputs are logged with timestamps.
- This confirms that the API and monitoring logic are functioning correctly.

