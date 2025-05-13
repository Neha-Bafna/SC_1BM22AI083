import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

df = pd.read_csv('/content/spam_ham_dataset.csv')

X = df['text']         # raw subject text
y = df['label_num']    # 0 for ham, 1 for spam

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
X_raw_train, X_raw_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = BaggingClassifier(estimator=DecisionTreeClassifier(), 
                          n_estimators=10, 
                          oob_score=True, 
                          random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"\n✅ Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"✅ OOB Score (Estimate of Generalization Accuracy): {model.oob_score_:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# Confusion Matrix
print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Sample Predictions (showing text, actual and predicted labels)
print("\n🔍 Sample Predictions (Actual vs Predicted):")
label_map = {0: "Ham", 1: "Spam"}
for i in range(10):
    actual = label_map[y_test.iloc[i]]
    predicted = label_map[y_pred[i]]
    text_sample = X_raw_test.iloc[i][:60].replace('\n', ' ')
    print(f"Text: {text_sample:60} | Actual: {actual:4} | Predicted: {predicted:4}")

print("\n📊 Accuracy of Individual Base Estimators on Test Set:")
for i, estimator in enumerate(model.estimators_):
    pred_i = estimator.predict(X_test)
    acc_i = accuracy_score(y_test, pred_i)
    print(f"Estimator {i+1:2}: Accuracy = {acc_i:.4f}")
