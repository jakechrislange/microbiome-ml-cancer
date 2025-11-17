from sklearn.linear_model import LogisticRegression
from processing import get_dataset
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss, recall_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

# Get dataset and split between train and test
X, y, class_names, X_train, X_test, y_train, y_test, feature_names = get_dataset()

scaler = StandardScaler()

# Scale and immediately convert to DataFrames with feature names
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_names,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_names,
    index=X_test.index
)


# Define parameter grid for Logistic Regression
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['saga'],  # saga supports all penalties
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # only used with elasticnet
    'max_iter': [1000]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    LogisticRegression(random_state=42, class_weight='balanced'),
    param_grid,
    cv=cv,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)

# Fit the grid search to the training data
grid_search.fit(X_train_scaled, y_train)

print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validated score:", grid_search.best_score_)

best_params = grid_search.best_params_

n_bootstrap = 1000
k_folds = 5
bootstrap_scores = []
all_conf_matrices = []

for boot_iter in range(n_bootstrap):
    print(f"Bootstrap iteration {boot_iter+1}/{n_bootstrap}")
    # Create a bootstrap resample from training
    X_boot, y_boot = resample(X_train_scaled, y_train, random_state=boot_iter, stratify=y_train)
    
    # Initialize cross-validation
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=boot_iter)
    fold_scores = []
    
    for train_idx, val_idx in kf.split(X_boot, y_boot):
        X_tr, X_val = X_boot.iloc[train_idx], X_boot.iloc[val_idx]
        y_tr, y_val = y_boot[train_idx], y_boot[val_idx]
        
        # Fit Logistic Regression on the bootstrap fold
        lr = LogisticRegression(**best_params, random_state=boot_iter, class_weight='balanced')
        lr.fit(X_tr, y_tr)

        # Predict and score
        y_val_proba = lr.predict_proba(X_val)[:, 1]
        y_val_pred = lr.predict(X_val)
        try:
            score = roc_auc_score(y_val, y_val_proba)
        except ValueError:
            continue
        fold_scores.append(score)

        cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1])
        all_conf_matrices.append(cm)
    
    # Store the mean cross-validated value for this bootstrap iteration
    if fold_scores:
        bootstrap_scores.append(np.mean(fold_scores))

# Empirical confidence interval
lower = np.percentile(bootstrap_scores, 2.5)
upper = np.percentile(bootstrap_scores, 97.5)
print(f"Mean bootstrapped CV ROC AUC: {np.mean(bootstrap_scores):.4f}")
print(f"95% confidence interval: [{lower:.4f}, {upper:.4f}]")

# Calculate average confusion matrix
mean_conf_matrix = np.mean(all_conf_matrices, axis=0)
print("Average Confusion Matrix:\n", mean_conf_matrix)

# Normalize rows
row_sums = mean_conf_matrix.sum(axis=1, keepdims=True)
norm_conf_matrix = np.divide(mean_conf_matrix, row_sums, where=row_sums!=0)

# Plot as heatmap
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(norm_conf_matrix, annot=True, fmt=".2f",
            xticklabels=["COAD", "READ"],
            yticklabels=["COAD", "READ"],
            cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.title('Bootstrap CV Confusion Matrix')
plt.show()


# HOLD OUT SET: 
best_model = LogisticRegression(**best_params, random_state=42, class_weight='balanced')
best_model.fit(X_train_scaled, y_train)

# Predict on holdout test set
y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = best_model.predict(X_test_scaled)

# Compute metrics on test set
test_roc_auc = roc_auc_score(y_test, y_test_proba)
test_cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

print(f"Test set ROC AUC: {test_roc_auc:.4f}")
print("Test set Confusion Matrix:")
print(test_cm)

# Normalize test confusion matrix for plotting
test_cm_norm = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]

# Plot heatmap
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(test_cm_norm, annot=True, fmt=".2f",
            xticklabels=["COAD", "READ"],
            yticklabels=["COAD", "READ"],
            cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.title('Holdout Test Set Confusion Matrix')
plt.show()


import shap
print("STARTING SHAP")

# ============ SHAP ANALYSIS ============
# For linear models, use LinearExplainer (fast and exact)
explainer = shap.LinearExplainer(best_model, X_train_scaled)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_scaled)

# For binary classification, shap_values might be a list [class_0, class_1]
# We want the SHAP values for the positive class (READ)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Plot summary
shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (READ vs COAD)")
plt.tight_layout()
plt.show()

# Print top 10 most important features
feature_importance = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(feature_importance)[::-1][:10]

print("Top 10 most important genera:")
for i, idx in enumerate(top_idx, 1):
    genus_name = feature_names[idx]
    importance = feature_importance[idx]
    print(f"{i}. {genus_name}: {importance:.4f}")

# For each top feature, calculate mean SHAP value (not absolute)
for idx in top_idx:
    genus_name = feature_names[idx]
    mean_shap = shap_values[:, idx].mean()
    
    if mean_shap > 0:
        print(f"{genus_name}: Higher abundance → READ")
    else:
        print(f"{genus_name}: Higher abundance → COAD")


#### Calibration and Brier Score ####

brier = brier_score_loss(y_test, y_test_proba)
print(f"Brier score: {brier:.4f}")

fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_test_proba, n_bins=5)

plt.figure(figsize=(8,6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model calibration")
plt.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve (Reliability Diagram)")
plt.legend()
plt.grid(True)
plt.show()

# Apply threshold adjustment (GHOST method)

train_proba = best_model.predict_proba(X_train_scaled)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.05)
best_threshold = 0.5
best_score = 0

for thresh in thresholds:
    train_pred = (train_proba >= thresh).astype(int)
    sens = recall_score(y_train, train_pred, pos_label=1)  # Sensitivity
    spec = recall_score(y_train, train_pred, pos_label=0)  # Specificity
    hmean = 2 * (sens * spec) / (sens + spec + 1e-6)      # Harmonic mean
    
    if hmean > best_score:
        best_score = hmean
        best_threshold = thresh

print("Optimized threshold:", best_threshold)

test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
test_pred_ghost = (test_proba >= best_threshold).astype(int)

test_roc_auc = roc_auc_score(y_test, y_test_proba)
print(f"Test set ROC AUC (unaffected by GHOST): {test_roc_auc:.4f}")

test_cm = confusion_matrix(y_test, test_pred_ghost, labels=[0, 1])
print("Test set Confusion Matrix with GHOST:")
print(test_cm)

# Normalize test confusion matrix for plotting
test_cm_norm = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]

# Plot heatmap
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(test_cm_norm, annot=True, fmt=".2f",
            xticklabels=["COAD", "READ"],
            yticklabels=["COAD", "READ"],
            cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.title('Holdout Test Set Confusion Matrix after GHOST')
plt.show()

# Also show coefficients from the model itself
coefficients = best_model.coef_[0]
top_idx_coef = np.argsort(np.abs(coefficients))[::-1][:10]

print("\nTop 10 features by Logistic Regression coefficient magnitude:")
for i, idx in enumerate(top_idx_coef, 1):
    genus_name = feature_names[idx]
    coef = coefficients[idx]
    print(f"{i}. {genus_name}: {coef:.4f} ({'READ' if coef > 0 else 'COAD'})")