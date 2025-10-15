# Iris Flower Classification with Confusion Matrix Visualization
# Author: Nikhil Saklani

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Step 1: Load Dataset ---
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for better understanding
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target_names[y]
print("✅ Sample Data:")
print(df.head())

# --- Step 2: Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Feature Scaling ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Step 4: Train Models ---
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# --- Step 5: Evaluate Models ---
log_acc = accuracy_score(y_test, log_pred)
knn_acc = accuracy_score(y_test, knn_pred)

print("\n📊 Model Accuracy Comparison:")
print(f"Logistic Regression Accuracy: {log_acc:.2f}")
print(f"KNN Accuracy: {knn_acc:.2f}")

# --- Step 6: Select Best Model ---
if log_acc > knn_acc:
    print("\n🏆 Logistic Regression performed better!")
    best_pred = log_pred
    best_name = "Logistic Regression"
else:
    print("\n🏆 K-Nearest Neighbors performed better!")
    best_pred = knn_pred
    best_name = "K-Nearest Neighbors"

# --- Step 7: Classification Report ---
print(f"\nClassification Report ({best_name}):")
print(classification_report(y_test, best_pred, target_names=iris.target_names))

# --- Step 8: Confusion Matrix ---
cm = confusion_matrix(y_test, best_pred)
print("\nConfusion Matrix:\n", cm)

# --- Step 9: Visualize Confusion Matrix ---
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix - {best_name}')
plt.show()
