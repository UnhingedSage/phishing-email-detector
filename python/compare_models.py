import joblib

# Load both models
model_v1 = joblib.load("python/models/phishing_model.joblib")
model_v2 = joblib.load("python/models/phishing_model_v2.joblib")

# Emails we previously tested
test_emails = [
    (
        "Hello, we noticed a discrepancy in your recent account activity. "
        "Please review the attached instructions to confirm your details.",
        "phishing"
    ),
    (
        "Hello, your monthly bank statement is now available. "
        "You can view it by signing into your banking app as usual.",
        "legitimate"
    ),
    (
        "Dear student, your university email access requires renewal. "
        "Complete the verification form at the provided link to maintain access.",
        "phishing"
    ),
    (
        "Your account password was successfully changed. "
        "If you made this change, no further action is required.",
        "legitimate"
    ),
    (
        "Congratulations! You have been selected for a remote position. "
        "To complete onboarding, submit your banking details through our registration portal.",
        "phishing"
    )
]

correct_v1 = 0
correct_v2 = 0

print("\n===== MODEL COMPARISON =====")

for i, (email, actual) in enumerate(test_emails, start=1):

    prediction_v1 = model_v1.predict([email])[0]
    prediction_v2 = model_v2.predict([email])[0]

    correct_v1 += prediction_v1 == actual
    correct_v2 += prediction_v2 == actual

    print(f"\nEmail {i}")
    print(f"Actual: {actual}")
    print(f"V1 prediction: {prediction_v1}")
    print(f"V2 prediction: {prediction_v2}")

print("\n===== FINAL RESULTS =====")
print(f"V1 accuracy: {correct_v1}/{len(test_emails)}")
print(f"V2 accuracy: {correct_v2}/{len(test_emails)}")