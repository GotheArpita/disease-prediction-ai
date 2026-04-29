from flask import Flask, request, jsonify, render_template
import joblib
import json
import numpy as np
import os

app = Flask(__name__)

# Load model
model = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")

with open("model_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_cols"]
URGENCY = metadata["urgency_map"]

# 🔥 FULL ADVICE SYSTEM (disease + urgency based)
ADVICE = {
    "Asthma": {
        "High": "Severe breathing difficulty detected. Seek emergency medical care immediately.",
        "Medium": "Avoid triggers, rest, and consult a doctor if symptoms persist.",
        "Low": "Monitor breathing and avoid allergens or triggers."
    },
    "Allergy": {
        "High": "Severe allergic reaction possible. Seek urgent medical attention.",
        "Medium": "Avoid allergens and take precautions. Consult doctor if needed.",
        "Low": "Mild symptoms. Maintain hygiene and avoid triggers."
    },
    "Bronchitis": {
        "High": "Persistent breathing issues. Seek medical evaluation urgently.",
        "Medium": "Rest, hydrate, and monitor cough.",
        "Low": "Mild irritation. Stay warm and hydrated."
    },
    "Common Cold": {
        "High": "Unusual severity. Seek medical attention.",
        "Medium": "Rest, fluids, and monitor symptoms.",
        "Low": "Mild cold. Home care is sufficient."
    },
    "Food Poisoning": {
        "High": "Severe dehydration risk. Seek medical help immediately.",
        "Medium": "Stay hydrated and eat light food.",
        "Low": "Mild symptoms. Rest and hydration recommended."
    },
    "Gastritis": {
        "High": "Severe abdominal pain. Seek medical attention.",
        "Medium": "Avoid spicy food and eat light meals.",
        "Low": "Manage diet and avoid irritants."
    },
    "Influenza": {
        "High": "High fever and fatigue. Seek medical care.",
        "Medium": "Rest and increase fluid intake.",
        "Low": "Mild flu. Home care and rest."
    },
    "Migraine": {
        "High": "Severe headache. Consult doctor immediately.",
        "Medium": "Rest in a dark and quiet place.",
        "Low": "Mild headache. Avoid triggers."
    },
    "Sinusitis": {
        "High": "Severe sinus pressure. Seek medical attention.",
        "Medium": "Steam inhalation and hydration.",
        "Low": "Mild discomfort. Home remedies help."
    },
    "Ulcer": {
        "High": "Severe stomach pain. Immediate medical care needed.",
        "Medium": "Avoid acidic food and consult doctor.",
        "Low": "Manage diet and monitor symptoms."
    }
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
        top_indices = np.argsort(probs)[-3:][::-1]

        labels = ["Most Likely", "Possible", "Less Likely"]
        results = []

        for i, idx in enumerate(top_indices):
            disease = le.inverse_transform([idx])[0].strip()
            urgency = URGENCY.get(disease, "Medium")

            # 🔥 dynamic advice selection
            advice = ADVICE.get(disease, {}).get(urgency, "Consult a doctor.")

            results.append({
                "rank": labels[i],
                "disease": disease,
                "confidence": round(float(probs[idx]) * 100, 2),
                "urgency": urgency,
                "advice": advice
            })

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))