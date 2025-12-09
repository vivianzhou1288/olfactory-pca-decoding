import numpy as np
import random

np.random.seed(42)
random.seed(42)

import pandas as pd
from preprocessing import prepare_data, prepare_data_neural_only
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

print("KNN Classification — All Features")
print("=" * 80)

# Load data
X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_data()

# Train KNN (k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluation
print("\nClassification Report (All Features):")
print(classification_report(y_test, y_pred, target_names=['Incorrect', 'Correct']))

print("\nConfusion Matrix (All Features):")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nConfusion Matrix Interpretation:")
print(f"  True Negatives: {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}")
print(f"  True Positives: {cm[1,1]}")

# Performance metrics
train_acc = knn.score(X_train, y_train)
test_acc = knn.score(X_test, y_test)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print("\nPerformance Metrics (All Features)")
print("=" * 80)
print(f"Training Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2%} (± {cv_scores.std():.2%})")
print(f"Overfitting Check: {train_acc - test_acc:+.2%}")

print("\n\nKNN Classification — Neural-Only Features")
print("=" * 80)

# Load neural-only data
X_train_neural, X_test_neural, y_train_neural, y_test_neural, feature_cols_neural, scaler_neural = prepare_data_neural_only()

# Train
knn_neural = KNeighborsClassifier(n_neighbors=5)
knn_neural.fit(X_train_neural, y_train_neural)

# Predictions
y_pred_neural = knn_neural.predict(X_test_neural)

print("\nClassification Report (Neural-Only):")
print(classification_report(y_test_neural, y_pred_neural, target_names=['Incorrect', 'Correct']))

print("\nConfusion Matrix (Neural-Only):")
cm_neural = confusion_matrix(y_test_neural, y_pred_neural)
print(cm_neural)

# Performance metrics
train_acc_neural = knn_neural.score(X_train_neural, y_train_neural)
test_acc_neural = knn_neural.score(X_test_neural, y_test_neural)
cv_scores_neural = cross_val_score(knn_neural, X_train_neural, y_train_neural, cv=5)

print("\nPerformance Metrics (Neural-Only)")
print("=" * 80)
print(f"Training Accuracy: {train_acc_neural:.2%}")
print(f"Test Accuracy: {test_acc_neural:.2%}")
print(f"Cross-Validation Accuracy: {cv_scores_neural.mean():.2%} (± {cv_scores_neural.std():.2%})")
print(f"Overfitting Check: {train_acc_neural - test_acc_neural:+.2%}")

# side by side comparison

comparison = pd.DataFrame({
    'Metric': ['Test Accuracy', 'CV Accuracy', 'CV Std Dev', 'Overfitting'],
    'All Features': [
        f"{test_acc:.2%}",
        f"{cv_scores.mean():.2%}",
        f"{cv_scores.std():.2%}",
        f"{train_acc - test_acc:+.2%}",
    ],
    'Neural-Only': [
        f"{test_acc_neural:.2%}",
        f"{cv_scores_neural.mean():.2%}",
        f"{cv_scores_neural.std():.2%}",
        f"{train_acc_neural - test_acc_neural:+.2%}",
    ]
})

print("\nSide-by-side Comparison")
print("=" * 80)
print(comparison.to_string(index=False))