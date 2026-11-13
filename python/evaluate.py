import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load our original dataset
df = pd.read_csv("data/clean_validation_emails.csv")

# Load our trained model
model = joblib.load("python/models/phishing_model.joblib")

# Separate email messages and labels
X = df["Email Text"]
y = df["Email Type"].map({
    "Safe Email": "legitimate",
    "Phishing Email": "phishing"
})

# Make predictions
predictions = model.predict(X)

# Display evaluation results
print("\n===== INDEPENDENT DATASET EVALUATION =====")

print(f"Total emails tested: {len(df)}")

print(f"Accuracy: {accuracy_score(y, predictions) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y, predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y,
    predictions,
    labels=["legitimate", "phishing"]
))

# Display individual predictions
print("\n===== INDIVIDUAL PREDICTIONS =====")

for i in range(len(df)):
    print(f"\nEmail {i + 1}")
    print(f"Actual: {y.iloc[i]}")
    print(f"Predicted: {predictions[i]}")
    print(f"Correct: {y.iloc[i] == predictions[i]}")