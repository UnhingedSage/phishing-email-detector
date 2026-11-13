import sys
from pathlib import Path
import joblib

# Locate the trained model
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "phishing_model_v2.joblib"
)

def predict_email(email):
    model = joblib.load(MODEL_PATH)

    # Predict email category
    prediction = model.predict([email])[0]

    # Calculate phishing probability
    probabilities = model.predict_proba([email])[0]

    # Find the probability corresponding to phishing
    phishing_index = list(model.classes_).index("phishing")
    phishing_probability = probabilities[phishing_index] * 100

    return prediction.upper(), phishing_probability


if __name__ == "__main__":
    email = sys.stdin.read().strip()

    if not email:
        print("ERROR: Empty email")
        sys.exit(1)

    prediction, probability = predict_email(email)

    print(f"Prediction: {prediction}")
    print(f"Phishing Probability: {probability:.2f}%")

    if probability >= 80:
        print("Risk Level: HIGH")
    elif probability >= 50:
        print("Risk Level: MEDIUM")
    else:
        print("Risk Level: LOW")