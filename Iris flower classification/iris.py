# Iris Flower Classification
# Author: Nikhil Saklani

# --- Step 1: Import libraries ---
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Step 2: Load the dataset ---
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame (for better visualization)
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target_names[y]
print("✅ Sample Data:")
print(df.head())

# --- Step 3: Split the data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Feature scaling ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Step 5: Train models ---

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

# --- Step 6: Evaluate models ---
log_acc = accuracy_score(y_test, log_pred)
knn_acc = accuracy_score(y_test, knn_pred)

print("\n📊 Model Accuracy Comparison:")
print(f"Logistic Regression Accuracy: {log_acc:.2f}")
print(f"KNN Accuracy: {knn_acc:.2f}")

# --- Step 7: Best Model Summary ---
if log_acc > knn_acc:
    print("\n🏆 Logistic Regression performed better!")
else:
    print("\n🏆 K-Nearest Neighbors performed better!")

print("\nClassification Report (Best Model):")
best_pred = log_pred if log_acc > knn_acc else knn_pred
print(classification_report(y_test, best_pred, target_names=iris.target_names))

# --- Step 8: Confusion Matrix ---
cm = confusion_matrix(y_test, best_pred)
print("\nConfusion Matrix:\n", cm)
