import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocessing import prepare_data, prepare_data_neural_only

np.random.seed(42)
sns.set_style("whitegrid")

# 1. Get data and train models
print("\nStep 1: Load Data & Train Models")

X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_data_neural_only()

print(f"Loaded neural features ({len(feature_cols)}): {feature_cols}")

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 2. Logistic regression: feature importance
print("\nStep 2: Logistic Regression Feature Importance")

coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': log_reg.coef_[0],
    'Abs_Coefficient': np.abs(log_reg.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop predictive features (sorted by |coefficient|):")
print(coef_df.to_string(index=False))

# 3. PCA Variance
print("\nStep 3: PCA Variance Explained")

variance_explained = pca.explained_variance_ratio_ * 100 # Tells you how much of the dataset’s total variance is captured by each principal component
cumulative_variance = np.cumsum(variance_explained)

# Summarizes how much of the total structure in the data is captured by the first few components
print("\nVariance explained per PC:")
for i, v in enumerate(variance_explained):
    print(f"  PC{i+1}: {v:.2f}% (cumulative: {cumulative_variance[i]:.2f}%)")

print(f"\nTop 2 PCs explain {cumulative_variance[1]:.1f}% of variance.")
print(f"Top 3 PCs explain {cumulative_variance[2]:.1f}%.\n")

# PCA Loadings: tells how each original feature contributes to each principal component
print("Step 4: PCA Feature Loadings")

loadings = pca.components_
loadings_df = pd.DataFrame(
    loadings.T,
    columns=[f'PC{i+1}' for i in range(len(feature_cols))],
    index=feature_cols
)

print("\nFeature contributions to each PC:")
print(loadings_df.to_string(float_format=lambda x: f"{x:+.3f}"))

print("\nDominant feature for each PC:")
for i in range(len(loadings)):
    idx = np.argmax(np.abs(loadings[i]))
    print(f"  PC{i+1}: {feature_cols[idx]} (loading {loadings[i, idx]:+.3f})")
print()

# Comparison
print("Step 5: LogReg vs PCA PC1 Comparison")

comparison_df = pd.DataFrame({
    'Feature': feature_cols,
    'LogReg_Coef': log_reg.coef_[0],
    'LogReg_Importance': np.abs(log_reg.coef_[0]),
    'PC1_Loading': loadings[0],
    'PC1_Importance': np.abs(loadings[0])
})

comparison_df['LogReg_Rank'] = comparison_df['LogReg_Importance'].rank(ascending=False)
comparison_df['PC1_Rank'] = comparison_df['PC1_Importance'].rank(ascending=False)

print("\nPredictive power vs variance direction:")
print(comparison_df[['Feature', 'LogReg_Coef', 'LogReg_Rank',
                     'PC1_Loading', 'PC1_Rank']]
      .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

#6. Predictive Performance
print("\nStep 6: Predictive Performance")

y_pred_logreg = log_reg.predict(X_test)
acc_logreg = accuracy_score(y_test, y_pred_logreg)

results = []
for n_pcs in range(1, len(feature_cols) + 1):
    lr_pca = LogisticRegression(random_state=42, max_iter=1000)
    lr_pca.fit(X_train_pca[:, :n_pcs], y_train)
    y_pred_pca = lr_pca.predict(X_test_pca[:, :n_pcs])
    results.append({
        'n_components': n_pcs,
        'test_accuracy': accuracy_score(y_test, y_pred_pca),
        'variance_captured': cumulative_variance[n_pcs-1]
    })

results_df = pd.DataFrame(results)

print("\nLogistic Regression using PCA components:")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

best_pca = results_df.loc[results_df['test_accuracy'].idxmax()]

print(f"\nOriginal feature accuracy: {acc_logreg:.2%}")
print(f"Best PCA model accuracy: {best_pca['test_accuracy']:.2%} "
      f"using {best_pca['n_components']} PCs "
      f"({best_pca['variance_captured']:.1f}% variance)\n")

# 7. Data Insights
print("\nStep 7: Data Insights")

comparison_df['Rank_Diff'] = np.abs(comparison_df['LogReg_Rank'] - comparison_df['PC1_Rank'])
comparison_df_sorted = comparison_df.sort_values('Rank_Diff', ascending=False)

print("\nFeatures with biggest disagreement:")
print(comparison_df_sorted[['Feature','LogReg_Rank','PC1_Rank','Rank_Diff']]
      .to_string(index=False))

top = comparison_df_sorted.iloc[0]

# Plot 1: Logistic Regression Coefficients
fig = plt.figure(figsize=(16, 10))

ax1 = plt.subplot(2, 3, 1)
colors = ['red' if c < 0 else 'blue' for c in log_reg.coef_[0]]
ax1.barh(feature_cols, log_reg.coef_[0], color=colors, alpha=0.7)
ax1.axvline(x=0, color='black', linewidth=0.8)
ax1.set_xlabel('Coefficient Value')
ax1.set_ylabel('Feature')
ax1.set_title('LogReg Coefficients')
ax1.grid(True, alpha=0.3)

# Plot 2: LogReg vs PC1 abs values
ax4 = plt.subplot(2, 3, 4)
x = np.arange(len(feature_cols))
width = 0.35
ax4.bar(x - width/2, comparison_df['LogReg_Importance'], width=width, label='LogReg')
ax4.bar(x + width/2, comparison_df['PC1_Importance'], width=width, label='PC1')
ax4.set_xticks(x)
ax4.set_xticklabels(feature_cols, rotation=45, ha='right')
ax4.set_xlabel('Feature')
ax4.set_ylabel('Absolute Importance')
ax4.set_title('Features Importance Comparison')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()