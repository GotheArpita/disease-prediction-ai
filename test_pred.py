import joblib, json, numpy as np
model = joblib.load('disease_model.pkl')
le = joblib.load('label_encoder.pkl')
with open('model_metadata.json') as f:
    metadata = json.load(f)
FEATURES = metadata['feature_cols']

def pred(age, symptoms):
    input_data = [0] * len(FEATURES)
    for i, feature in enumerate(FEATURES):
        if feature == 'Age': input_data[i] = age
        elif feature in symptoms: input_data[i] = 1
    pred = model.predict(np.array([input_data]))[0]
    preds_proba = model.predict_proba(np.array([input_data]))[0]
    res_proba = {le.inverse_transform([i])[0]: round(p*100, 2) for i, p in enumerate(preds_proba)}
    print(f'Age: {age}, Symptoms: {symptoms}')
    print(f'  => {le.inverse_transform([pred])[0]}')
    print(f'  => {dict(sorted(res_proba.items(), key=lambda item: item[1], reverse=True)[:3])}')

pred(25, ['fever'])
pred(25, ['fever', 'muscle pain', 'fatigue'])
pred(25, ['fever', 'muscle pain', 'fatigue', 'cough'])
pred(45, ['fever', 'fatigue'])
pred(45, ['fever', 'body pain', 'fatigue'])
