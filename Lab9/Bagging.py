import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load dataset
df = pd.read_csv("spam.csv")

# Extract features and labels
X = df['text']         # raw subject text
y = df['label_num']    # 0 for ham, 1 for spam

# Convert text to numerical features
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data - for vectorized data and raw text (for display)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
X_raw_train, X_raw_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize BaggingClassifier (with OOB scoring)
model = BaggingClassifier(estimator=DecisionTreeClassifier(),
                          n_estimators=10,
                          oob_score=True,
                          random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Overall Accuracy and OOB Score
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"OOB Score (Estimate of Generalization Accuracy): {model.oob_score_:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Sample Predictions (showing text, actual and predicted labels)
print("\nSample Predictions (Actual vs Predicted):")
label_map = {0: "Ham", 1: "Spam"}
for i in range(10):
    actual = label_map[y_test.iloc[i]]
    predicted = label_map[y_pred[i]]
    text_sample = X_raw_test.iloc[i][:60].replace('\n', ' ')
    print(f"Text: {text_sample:60} | Actual: {actual:4} | Predicted: {predicted:4}")

# Individual base estimator performance
print("\nAccuracy of Individual Base Estimators on Test Set:")
for i, estimator in enumerate(model.estimators_):
    pred_i = estimator.predict(X_test)
    acc_i = accuracy_score(y_test, pred_i)
    print(f"Estimator {i+1:2}: Accuracy = {acc_i:.4f}")
