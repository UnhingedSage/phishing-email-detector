import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load labelled email examples
data = pd.read_csv("data/emails.csv")

X = data["text"]
y = data["label"]

# Build a machine-learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train on the examples
model.fit(X, y)

# Save the trained model
joblib.dump(model, "python/models/phishing_model.joblib")

print("Model trained successfully!")
print(f"Training examples: {len(data)}")
print("Model saved to python/models/phishing_model.joblib")