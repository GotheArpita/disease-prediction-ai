import pandas as pd
import numpy as np
import joblib
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

print("="*60)
print("🔥 HIGH ACCURACY MEDICAL MODEL (XGBOOST)")
print("="*60)

DISEASES = [
    "Allergy", "Asthma", "Bronchitis", "Common Cold",
    "Food Poisoning", "Gastritis", "Influenza",
    "Migraine", "Sinusitis", "Ulcer"
]

SYMPTOMS = [
    "sneezing","runny nose","rash","shortness of breath","wheezing",
    "chest tightness","cough","mucus production","fever","sore throat",
    "nausea","vomiting","diarrhea","abdominal pain","appetite loss",
    "headache","muscle pain","fatigue","sweating","chills",
    "facial pressure","nasal congestion","sensitivity to light",
    "sensitivity to sound","throbbing pain","bloating","heartburn",
    "burping","back pain","dizziness","joint pain","eye irritation"
]

# 🔥 STRONG MEDICAL PROFILES
PROFILE = {
    "Bronchitis": ["cough","mucus production","chest tightness"],
    "Influenza": ["fever","fatigue","muscle pain"],
    "Migraine": ["headache","sensitivity to light","throbbing pain"],
    "Food Poisoning": ["vomiting","diarrhea","nausea"],
    "Asthma": ["shortness of breath","wheezing","chest tightness"],
    "Sinusitis": ["facial pressure","nasal congestion","headache"],
    "Gastritis": ["abdominal pain","bloating","heartburn"],
    "Ulcer": ["abdominal pain","heartburn","appetite loss"],
    "Allergy": ["sneezing","runny nose","rash"],
    "Common Cold": ["cough","sore throat","runny nose"]
}

rows = []
N = 900

for disease in DISEASES:
    for _ in range(N):

        age = np.random.randint(10, 70)
        symptoms = np.zeros(len(SYMPTOMS))

        strong = PROFILE[disease]

        # 🔥 Always include strong symptoms
        for s in strong:
            symptoms[SYMPTOMS.index(s)] = 1

        # 🔥 Add controlled secondary symptoms
        secondary = np.random.choice(SYMPTOMS, 3, replace=False)
        for s in secondary:
            if s not in strong:
                symptoms[SYMPTOMS.index(s)] = np.random.choice([0,1], p=[0.7,0.3])

        rows.append([age, disease] + list(symptoms))

df = pd.DataFrame(rows, columns=["Age","Disease"] + SYMPTOMS)

print("📊 Dataset:", df.shape)

X = df[["Age"] + SYMPTOMS].values
y = df["Disease"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# 🔥 XGBOOST TUNED
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy:", round(acc*100,2), "%")

print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save
joblib.dump(model, "disease_model.pkl")
joblib.dump(le, "label_encoder.pkl")

URGENCY = {
    "Allergy":"Low",
    "Asthma":"High",
    "Bronchitis":"Medium",
    "Common Cold":"Low",
    "Food Poisoning":"Medium",
    "Gastritis":"Medium",
    "Influenza":"Medium",
    "Migraine":"Medium",
    "Sinusitis":"Low",
    "Ulcer":"High"
}

metadata = {
    "feature_cols": ["Age"] + SYMPTOMS,
    "classes": list(le.classes_),
    "urgency_map": URGENCY
}

with open("model_metadata.json","w") as f:
    json.dump(metadata,f,indent=2)

print("\n✅ DONE — HIGH ACCURACY MODEL READY")