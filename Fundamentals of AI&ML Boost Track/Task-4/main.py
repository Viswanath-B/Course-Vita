from flask import Flask, request, jsonify
import joblib
import numpy as np
import datetime

app = Flask(__name__)

model = joblib.load("ecg_model.pkl")
MODEL_VERSION = "1.0-ECG"

# Determine expected features safely
if hasattr(model, "n_features_in_"):
    EXPECTED_FEATURES = model.n_features_in_
elif isinstance(model, np.ndarray):
    EXPECTED_FEATURES = model.shape[1] if model.ndim > 1 else model.shape[0]
else:
    EXPECTED_FEATURES = None

@app.route("/")
def home():
    return "ECG Signal Classification API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    data = list(data)

    if EXPECTED_FEATURES is not None:
        if len(data) < EXPECTED_FEATURES:
            data = data + [0] * (EXPECTED_FEATURES - len(data))
        else:
            data = data[:EXPECTED_FEATURES]

    data = np.array(data).reshape(1, -1)

    # Predict
    if hasattr(model, "predict"):
        prediction = model.predict(data)[0]
    else:
        prediction = "Invalid model"

    with open("ecg_prediction_logs.txt", "a") as f:
        f.write(str(datetime.datetime.now()) + "," + str(prediction) + "\n")

    return jsonify({
        "model_version": MODEL_VERSION,
        "ecg_class": str(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)
