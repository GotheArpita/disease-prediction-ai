from flask import Flask, request, jsonify, render_template
import joblib
import json
import numpy as np

app = Flask(__name__)

# Load model and files
model = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")

with open("model_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_cols"]
URGENCY = metadata["urgency_map"]

# Home route (for browser test)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/checker")
def checker():
    print("Rendering checker page")
    return render_template("checker.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    age = data["age"]
    symptoms = data["symptoms"]

    # Create input vector
    input_data = [0] * len(FEATURES)

    for i, feature in enumerate(FEATURES):
        if feature == "Age":
            input_data[i] = age
        elif feature in symptoms:
            input_data[i] = 1

    input_array = np.array([input_data])

    # Predict disease
    pred = model.predict(input_array)[0]
    disease = le.inverse_transform([pred])[0]

    # Get urgency
    urgency = URGENCY[disease]

    return jsonify({
        "disease": disease,
        "urgency": urgency["label"],
        "advice": urgency["advice"]
    })

if __name__ == "__main__":
    app.run(debug=True)