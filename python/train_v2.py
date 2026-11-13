import pandas as pd
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load both training datasets
original = pd.read_csv("data/emails.csv")
additional = pd.read_csv("data/additional_training_emails.csv")

# 2. Combine the datasets
data = pd.concat([original, additional], ignore_index=True)

# 3. Remove missing values and duplicate messages
data = data.dropna(subset=["text", "label"])
data["text"] = data["text"].astype(str).str.strip()
data["label"] = data["label"].astype(str).str.strip().str.lower()

data = data.drop_duplicates(subset=["text"])

# 4. Separate features and labels
X = data["text"]
y = data["label"]

print("===== MODEL V2 TRAINING =====")
print("Training examples:", len(data))
print("\nClass distribution:")
print(y.value_counts())

# 5. Create the ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# 6. Train the model
model.fit(X, y)

# 7. Save separately from the original model
model_path = Path("python/models/phishing_model_v2.joblib")
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_path)

print("\nModel V2 trained successfully!")
print("Saved to:", model_path)