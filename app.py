from flask import Flask, request, jsonify, render_template
import joblib
import json
import numpy as np
import os  # ✅ FIXED (moved here)

app = Flask(__name__)

# Load model
model = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")

with open("model_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_cols"]
URGENCY = metadata["urgency_map"]

# ✅ Advice mapping
ADVICE = {
    "Allergy": "Take antihistamines and avoid allergens.",
    "Asthma": "Use inhaler and seek medical help if severe.",
    "Bronchitis": "Rest, hydrate, and avoid cold air.",
    "Common Cold": "Rest and drink fluids.",
    "Food Poisoning": "Stay hydrated and eat light food.",
    "Gastritis": "Avoid spicy food and eat light meals.",
    "Influenza": "Rest, fluids, and medication if needed.",
    "Migraine": "Rest in dark room and avoid triggers.",
    "Sinusitis": "Steam inhalation and hydration.",
    "Ulcer": "Consult doctor and avoid acidic food."
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/checker")
def checker():
    return render_template("checker.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        age = data.get("age", 0)
        symptoms = data.get("symptoms", [])
        symptoms = [s.lower().strip() for s in symptoms]

        input_data = [0] * len(FEATURES)
        feature_map = [f.lower().strip() for f in FEATURES]

        for i, f in enumerate(feature_map):
            if f == "age":
                input_data[i] = age
            elif f in symptoms:
                input_data[i] = 1

        probs = model.predict_proba([input_data])[0]
        top = np.argsort(probs)[-3:][::-1]

        best_idx = top[0]
        disease = le.inverse_transform([best_idx])[0].strip()

        return jsonify({
            "disease": disease,
            "confidence": round(float(probs[best_idx]) * 100, 2),
            "urgency": URGENCY.get(disease, "Unknown"),
            "advice": ADVICE.get(disease, "Consult a doctor.")
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ FINAL RUN CONFIG (RENDER FIX)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))