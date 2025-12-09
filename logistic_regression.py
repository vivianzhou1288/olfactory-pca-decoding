import numpy as np
import random

np.random.seed(42)
random.seed(42)

import pandas as pd
from preprocessing import prepare_data, prepare_data_neural_only
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Get preprocessed data
X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_data()

# Train logistic regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

# Evaluate
y_pred = log_reg.predict(X_test)

print("Logistic Regression with Balanced Class Weights")
print("=" * 60)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Incorrect', 'Correct']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nConfusion Matrix Interpretation:")
print(f"  True Negatives (Incorrect predicted as Incorrect): {cm[0,0]}")
print(f"  False Positives (Incorrect predicted as Correct): {cm[0,1]}")
print(f"  False Negatives (Correct predicted as Incorrect): {cm[1,0]}")
print(f"  True Positives (Correct predicted as Correct): {cm[1,1]}")

# Performance metrics
train_acc = log_reg.score(X_train, y_train)
test_acc = log_reg.score(X_test, y_test)
cv_scores = cross_val_score(log_reg, X_train, y_train, cv=5)

print("\nPerformance Metrics")
print("=" * 60)
print(f"Training Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")
print(f"Cross-Validation Accuracy: {cv_scores.mean():.2%} (± {cv_scores.std():.2%})")
print(f"Overfitting Check: {train_acc - test_acc:+.2%}")

# Feature importance (coefficients)
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': log_reg.coef_[0],
    'Abs_Coefficient': np.abs(log_reg.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature importance (coefficients)")
print("=" * 60)
print(coef_df[['Feature', 'Coefficient']])
print(f"\nIntercept: {log_reg.intercept_[0]:.4f}")

# Interpretation
print("Intepretation")
print("=" * 60)
print("Positive coefficient → feature increases P(Correct)")
print("Negative coefficient → feature decreases P(Correct)")
print("Larger |coefficient| → stronger effect")

top_feature = coef_df.iloc[0]
print(f"\nMost important feature: {top_feature['Feature']}")
print(f"  Coefficient: {top_feature['Coefficient']:.4f}")
if top_feature['Coefficient'] > 0:
    print(f"  → Higher {top_feature['Feature']} predicts MORE correct trials")
else:
    print(f"  → Higher {top_feature['Feature']} predicts FEWER correct trials")

print("Comparing neural features only")
print("=" * 80)

# Get neural only data
X_train_neural, X_test_neural, y_train_neural, y_test_neural, feature_cols_neural, scaler_neural = prepare_data_neural_only()

# Training
log_reg_neural = LogisticRegression(random_state=42, max_iter=1000)
log_reg_neural.fit(X_train_neural, y_train_neural)

# Evaluate
y_pred_neural = log_reg_neural.predict(X_test_neural)

print("\nClassification Report (Neural-Only):")
print(classification_report(y_test_neural, y_pred_neural, target_names=['Incorrect', 'Correct']))

print("\nConfusion Matrix (Neural-Only):")
cm_neural = confusion_matrix(y_test_neural, y_pred_neural)
print(cm_neural)

# Performance metrics
train_acc_neural = log_reg_neural.score(X_train_neural, y_train_neural)
test_acc_neural = log_reg_neural.score(X_test_neural, y_test_neural)
cv_scores_neural = cross_val_score(log_reg_neural, X_train_neural, y_train_neural, cv=5)

print("\nPerformance Metrics (Neural-Only)")
print("=" * 60)
print(f"Training Accuracy: {train_acc_neural:.2%}")
print(f"Test Accuracy: {test_acc_neural:.2%}")
print(f"Cross-Validation Accuracy: {cv_scores_neural.mean():.2%} (± {cv_scores_neural.std():.2%})")
print(f"Overfitting Check: {train_acc_neural - test_acc_neural:+.2%}")

# Feature importance
coef_df_neural = pd.DataFrame({
    'Feature': feature_cols_neural,
    'Coefficient': log_reg_neural.coef_[0],
    'Abs_Coefficient': np.abs(log_reg_neural.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance (Neural-Only)")
print("=" * 60)
print(coef_df_neural[['Feature', 'Coefficient']].to_string(index=False))
print(f"\nIntercept: {log_reg_neural.intercept_[0]:.4f}")

# Interpretation
print("\nInterpretation (Neural-Only)")
print("=" * 60)
for idx, row in coef_df_neural.iterrows():
    feat = row['Feature']
    coef = row['Coefficient']
    if coef > 0:
        print(f"  ↑ {feat:20s} → ↑ P(Correct)  (coef: {coef:+.3f})")
    else:
        print(f"  ↓ {feat:20s} → ↑ P(Correct)  (coef: {coef:+.3f})")

# Side-by-side Comparison
print("\nSide-by-side Comparison")
print("=" * 80)

# Extract neural coefficients from the "All Features" model
theta_idx = feature_cols.index('theta_frequency')
beta_idx = feature_cols.index('beta_frequency')
gamma_idx = feature_cols.index('gamma_frequency')
noise_idx = feature_cols.index('noise_level')

# Get actual coefficients from "All Features" model
theta_coef_all = log_reg.coef_[0][theta_idx]
beta_coef_all = log_reg.coef_[0][beta_idx]
gamma_coef_all = log_reg.coef_[0][gamma_idx]
noise_coef_all = log_reg.coef_[0][noise_idx]

# Get neural coefficients from "Neural-Only" model
theta_coef_neural = log_reg_neural.coef_[0][0]  # theta is first
beta_coef_neural = log_reg_neural.coef_[0][1]   # beta is second
gamma_coef_neural = log_reg_neural.coef_[0][2]  # gamma is third
noise_coef_neural = log_reg_neural.coef_[0][3]  # noise is fourth

comparison = pd.DataFrame({
    'Metric': ['Test Accuracy', 'CV Accuracy', 'CV Std Dev', 'Overfitting', 
               'Theta Coef', 'Beta Coef', 'Gamma Coef', 'Noise Coef'],
    'All Features': [
        f"{test_acc:.2%}",
        f"{cv_scores.mean():.2%}",
        f"{cv_scores.std():.2%}",
        f"{train_acc - test_acc:+.2%}",
        f"{theta_coef_all:+.3f}",
        f"{beta_coef_all:+.3f}",
        f"{gamma_coef_all:+.3f}",
        f"{noise_coef_all:+.3f}"
    ],
    'Neural-Only': [
        f"{test_acc_neural:.2%}",
        f"{cv_scores_neural.mean():.2%}",
        f"{cv_scores_neural.std():.2%}",
        f"{train_acc_neural - test_acc_neural:+.2%}",
        f"{theta_coef_neural:+.3f}",
        f"{beta_coef_neural:+.3f}",
        f"{gamma_coef_neural:+.3f}",
        f"{noise_coef_neural:+.3f}"
    ]
})

print(comparison.to_string(index=False))