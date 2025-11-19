from sklearn.neural_network import MLPClassifier
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

from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)


# Define parameter grid and perform grid search for hyperparameter selection
param_grid = {
    'hidden_layer_sizes': [(5,), (10, 5), (10,), (20,), (10,10)],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.0001, 0.001, 0.01],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    #'learning_rate': ['constant', 'adaptive'] # only used with 'sgd' solver
}
# param_grid = {
#     'hidden_layer_sizes': [
#         (16,), (32,), 
#         (16, 16), (32, 16), (32, 64, 64, 16), (32, 64, 16)
#     ],
#     'alpha': [1e-4, 3e-4, 1e-3],
#     'activation': ['relu', 'tanh'],
#     'solver': ['adam', 'lbfgs'],
#     'learning_rate_init': [1e-2, 1e-3, 3e-4],  # stable for Adam
# }


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    MLPClassifier(max_iter=2000, early_stopping=True, random_state=42),
    param_grid,
    cv=cv,
    scoring='roc_auc',
    verbose=0,
    n_jobs=-1
)

# Fit the grid search to the training data
grid_search.fit(X_train_scaled, y_train)

print("Best hyperparameters:", grid_search.best_params_)
best_hidden_layers = grid_search.best_params_['hidden_layer_sizes']
best_alpha = grid_search.best_params_['alpha']
best_lr_init = grid_search.best_params_['learning_rate_init']
best_activation = grid_search.best_params_['activation']
best_solver = grid_search.best_params_['solver']
#best_learning_rate = grid_search.best_params_['learning_rate']
print("Best hidden_layer_sizes:", best_hidden_layers)
print("Best alpha:", best_alpha)
print("Best learning_rate_init:", best_lr_init)
print("Best activation:", best_activation)
print("Best solver:", best_solver)
#print("Best learning_rate:", best_learning_rate)
print("Best cross-validated score:", grid_search.best_score_)


#exit(1)
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

        fold_sample_weights = compute_sample_weight('balanced', y_tr)
        
        # Fit MLP on the bootstrap fold
        mlp = MLPClassifier(
            hidden_layer_sizes=best_hidden_layers,
            alpha=best_alpha,
            learning_rate_init=best_lr_init,
            activation=best_activation,
            solver=best_solver,
            #learning_rate=best_learning_rate,
            max_iter=2000,
            early_stopping=True,
            random_state=boot_iter
        )
        mlp.fit(X_tr, y_tr)

        # Predict and score (ROC AUC recommended for imbalance)
        y_val_proba = mlp.predict_proba(X_val)[:, 1]
        y_val_pred = mlp.predict(X_val)
        try:
            score = roc_auc_score(y_val, y_val_proba)
        except ValueError:
            continue  # Skip if ROC AUC can't be computed
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
best_model = MLPClassifier(
    hidden_layer_sizes=best_hidden_layers,
    alpha=best_alpha,
    learning_rate_init=best_lr_init,
    activation=best_activation,
    solver=best_solver,
    #learning_rate=best_learning_rate,
    max_iter=2000,
    early_stopping=True,
    random_state=boot_iter
)
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

# Define wrapper that converts arrays to DataFrames
def model_predict(x):
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x, columns=feature_names)
    return best_model.predict_proba(x)[:, 1]

# Use SCALED data for background
background = shap.sample(X_train_scaled, 100, random_state=42)

# Create explainer with wrapper
explainer = shap.KernelExplainer(
    model=model_predict,
    data=background
)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_scaled)

# Plot with X_test DataFrame (has feature names) but values match X_test_scaled
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

# Apply threshold adjust to try accounting for class imbalance -- GHOST: https://pubmed.ncbi.nlm.nih.gov/34100609/

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

# With threshold adjustment we see near-chance results, highlighting the similarities between READ and COAD
# and the difficulties in distinguishing them. This is important to keep and report on.

# With the GHOST determined threshold we get near random results. This gives us a Confusion Matrix
# that's more accurate considering the class imbalance