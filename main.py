import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, make_scorer, recall_score, roc_auc_score, accuracy_score
import numpy as np
from train import train
from train_with_bootstrap import train_with_bootstrap
from train_with_boot_and_smote import train_with_boot_and_smote

from train_hyperparameter_optim import grid_search

# Load the CSV you saved previously
dataset = pd.read_csv("TCMA_Genus_Processed/final.csv")

# print(dataset.head())

start_idx = dataset.columns.get_loc("HistologicalType") + 1
X = dataset.iloc[:, start_idx:] 
y = dataset["project"]  



#train()  # Train the model using only COAD and READ samples
# print("Training with bootstrap for different input dimensions:\n")
# print("Input Dim: 32:\n")
# train_with_bootstrap(second_dim=32)
#print("Second Dim: 64:\n")
#train_with_bootstrap(second_dim=64)

#train_with_boot_and_smote(second_dim = 64, use_smote=True, smote_strategy='auto')
#grid_search()

# print("Second Dim: 128:\n")
train_with_boot_and_smote(second_dim = 128, use_smote=True, smote_strategy='auto')
# print("Input Dim: 128:\n")
# train_with_bootstrap(second_dim=128)
# print("Input Dim: 256:\n")
# train_with_bootstrap(second_dim=256)


# # print("X shape:", X.shape)
# # print("y shape:", y.shape)

# # print(X.head())
# # print(y.head())


# #The RF models were trained and tested on separate, stratified sampling splits of 85% and 15% of the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y, shuffle=True)

# n_iterations = 1000
# auc_scores, accuracy_scores, sensitivity_scores, specificity_scores = [], [], [], []

# for i in range(n_iterations):
#     print("Jake: Iternation ", i)
#     # Bootstrap resample
#     sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
#     X_resample, y_resample = X_train.iloc[sample_idx], y_train.iloc[sample_idx]
    
#     # Reinitialize model each iteration
#     clf = MLPClassifier(
#         solver='adam',
#         alpha=1e-5,
#         hidden_layer_sizes=(100, 50, 25), # not specifying this got auc 0.924, 0.962
#         random_state=None,
#         learning_rate_init=0.001, 
#         max_iter=500
#     )
#     clf.fit(X_resample, y_resample)
    
#     # Evaluate on test set
#     y_pred = clf.predict(X_test)
#     y_prob = clf.predict_proba(X_test)
    
#     # Metrics (multiclass)
#     auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
#     acc = accuracy_score(y_test, y_pred)
#     sensitivity = recall_score(y_test, y_pred, average='macro')  # mean recall
#     specificity = recall_score(y_test, y_pred, average='macro')  # placeholder: same as recall for multiclass

#     auc_scores.append(auc)
#     accuracy_scores.append(acc)
#     sensitivity_scores.append(sensitivity)
#     specificity_scores.append(specificity)

# # Confidence intervals
# lower = np.percentile(auc_scores, 2.5)
# upper = np.percentile(auc_scores, 97.5)

# # Summary stats
# mean_auc = np.mean(auc_scores)
# std_dev_auc = np.std(auc_scores, ddof=1)
# mean_accuracy = np.mean(accuracy_scores)
# mean_sensitivity = np.mean(sensitivity_scores)
# mean_specificity = np.mean(specificity_scores)

# # Results
# print(f"Mean AUC (macro): {mean_auc:.3f}")
# print(f"95% CI for AUC: [{lower:.3f}, {upper:.3f}]")
# print(f"Std Dev AUC: {std_dev_auc:.4f}")
# print(f"Mean Accuracy: {mean_accuracy:.3f}")
# print(f"Mean Sensitivity (macro recall): {mean_sensitivity:.3f}")
# print(f"Mean Specificity (same as macro recall in multiclass): {mean_specificity:.3f}")
