import joblib
import json
import numpy as np

model = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")

with open("model_metadata.json") as f:
    metadata = json.load(f)

FEATURES = metadata["feature_cols"]

def pred(age, symptoms):
    input_data = [0]*len(FEATURES)

    for i,f in enumerate(FEATURES):
        if f == "Age":
            input_data[i] = age
        elif f in symptoms:
            input_data[i] = 1

    probs = model.predict_proba([input_data])[0]

    top = np.argsort(probs)[-3:][::-1]

    print("\n--- RESULT ---")
    for i in top:
        print(le.inverse_transform([i])[0], round(probs[i]*100,2),"%")

pred(25, ["fever","cough"])