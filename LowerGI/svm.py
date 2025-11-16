


from sklearn.svm import SVC
from processing import get_dataset
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Get dataset and split between train and test
X, y, class_names = get_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y, shuffle=True
)
feature_names = X_train.columns.tolist()

import pandas as pd

print("Class distribution:")
print(pd.Series(y_train).value_counts())
print(pd.Series(y_train).value_counts(normalize=True))


# Fit scaler ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the SAME scaler (fitted on train)
X_test_scaled = scaler.transform(X_test)


# Convert back to DataFrames with original column names
X_train = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)


#Define parameter grid and perfrom grid search for hyperparameter selection
param_grid = {
    'C': [0.01, 0.1, 1, 10, 25],
    'gamma': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, "scale", "auto"],
    'kernel': ['rbf']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    SVC(probability=True, class_weight='balanced'),
    param_grid,
    cv=cv,
    scoring='roc_auc',
    verbose=0,
    n_jobs=-1
)

# Fit the grid search to the training data
grid_search.fit(X_train_scaled, y_train)

print("Best hyperparameters:", grid_search.best_params_)
best_c = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']
print("Best C:", best_c)
print("Best gamma:", best_gamma)
print("Best cross-validated score:", grid_search.best_score_)


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
        X_tr, X_val = X_boot[train_idx], X_boot[val_idx]
        y_tr, y_val = y_boot[train_idx], y_boot[val_idx]
        
        # Fit SVM on the bootstrap fold
        svm = SVC(kernel='rbf', gamma=best_gamma, C=best_c, probability=True, class_weight='balanced')
        svm.fit(X_tr, y_tr)

        # Predict and score (ROC AUC recommended for imbalance)
        y_val_proba = svm.predict_proba(X_val)[:, 1]
        y_val_pred = svm.predict(X_val)
        try:
            score = roc_auc_score(y_val, y_val_proba)
        except ValueError:
            continue  # Skip if ROC AUC can't be computed
        fold_scores.append(score)

        cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1]) # or use class_names if you prefer
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

# Normalize rows (e.g., to get percentage per-class; each row sums to 1)
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
plt.show()


# HOLD OUT SET: 
best_model = SVC(kernel='rbf', gamma=best_gamma, C=best_c, probability=True, class_weight='balanced')
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

# ============ FIXED SHAP ANALYSIS ============

# Use SCALED data for background (not X_train)
background = shap.sample(X_train_scaled, 100, random_state=42)

# Create explainer
explainer = shap.KernelExplainer(
    model=lambda x: best_model.predict_proba(x)[:, 1],
    data=background
)

# Calculate SHAP values using SCALED test data
shap_values = explainer.shap_values(X_test_scaled)

# Plot with X_test DataFrame (has feature names) but values match X_test_scaled
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (READ vs COAD)")
plt.tight_layout()
plt.show()

shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Feature Impact")
plt.tight_layout()
plt.show()

# Force plot
shap.force_plot(
    explainer.expected_value, 
    shap_values[0], 
    X_test.iloc[0],  # Use iloc for safer indexing
    matplotlib=True,
    show=False
)
plt.title("SHAP Force Plot - First Test Sample")
plt.tight_layout()
plt.show()

# Waterfall plot
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0]
), show=False)
plt.tight_layout()
plt.show()