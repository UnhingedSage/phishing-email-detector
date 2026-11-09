import sys
from pathlib import Path
import joblib

# Location of my trained model
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "phishing_model.joblib"
)

def predict_email(email):
    # Load our trained machine-learning model
    model = joblib.load(MODEL_PATH)

    # Predict the email category
    prediction = model.predict([email])[0]

    return prediction.upper()


if __name__ == "__main__":
    email = sys.stdin.read()

    if not email.strip():
        print("ERROR: Empty email")
        sys.exit(1)

    print(predict_email(email))