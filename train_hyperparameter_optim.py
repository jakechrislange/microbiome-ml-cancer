import torch
from model import MLPClassifierDeepResidual
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import torch.nn.functional as F
import numpy as np
from itertools import combinations


def get_dataset():
    dataset = pd.read_csv("TCMA_Genus_Processed/final.csv")
    start_idx = dataset.columns.get_loc("HistologicalType") + 1
    X = dataset.iloc[:, start_idx:]
    y = dataset["project"].astype("category").cat.codes.values
    class_names = dataset["project"].astype("category").cat.categories
    return X, y, class_names


def compute_class_weights(y):
    """Compute inverse frequency class weights"""
    classes, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    return class_weights


def apply_smote(X, y, sampling_strategy="auto", k_neighbors=5):
    sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def train_single_fold(X_tr, y_tr, X_val, y_val, second_dim, num_classes, class_weights, device, batch_size=8, lr=1e-3, num_epoch=50, weight_decay=1e-4):
    """Train a single fold and return validation metrics"""
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    model = MLPClassifierDeepResidual(input_dim=X_tr.shape[1], second_dim=second_dim, num_classes=num_classes).to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epoch):
        model.train()
        for x_data, y_data in train_loader:
            x_data, y_data = x_data.to(device), y_data.to(device)
            pred = model(x_data)
            loss_val = loss_func(pred, y_data)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

    # Final validation evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for x_data, y_data in val_loader:
            x_data, y_data = x_data.to(device), y_data.to(device)
            pred = model(x_data)
            probs = F.softmax(pred, dim=1).cpu().numpy()
            labels = y_data.cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate accuracy
    val_acc = (all_preds.argmax(axis=1) == all_labels).mean()
    
    # Calculate AUC
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_preds[:, 1])
        else:
            auc = roc_auc_score(y_true=all_labels, y_score=all_preds, 
                               multi_class="ovo", average="macro")
    except ValueError:
        auc = float("nan")
    
    return val_acc, auc, all_preds, all_labels


def train_kfold(second_dim=128, batch_size=8, lr=1e-3, num_epoch=50, use_smote=False, smote_strategy="auto", smote_k_neighbors=5, weight_decay=1e-4):
    seed = 42
    k_folds = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA not available, using CPU")

    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y, class_names = get_dataset()
    num_classes = len(class_names)

    # Split once into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y, shuffle=True
    )

    class_weights = compute_class_weights(y_train).to(device)

    # K-Fold CV
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_acc, fold_auc = [], []
    fold_per_class_auc = {cls: [] for cls in class_names}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        #print(f"\n=== Fold {fold + 1}/{k_folds} ===")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Apply SMOTE safely within training fold
        if use_smote:
            X_tr, y_tr = apply_smote(X_tr, y_tr, sampling_strategy=smote_strategy, k_neighbors=smote_k_neighbors)
            print(f"SMOTE applied — training size: {len(y_tr)}")

        X_tr_tensor = torch.tensor(X_tr.values, dtype=torch.float32)
        y_tr_tensor = torch.tensor(y_tr, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        val_acc, auc, all_preds, all_labels = train_single_fold(
            X_tr_tensor, y_tr_tensor, X_val_tensor, y_val_tensor,
            second_dim, num_classes, class_weights, device, batch_size, lr, num_epoch, weight_decay
        )

        fold_acc.append(val_acc)
        fold_auc.append(auc)

        for i, cls in enumerate(class_names):
            try:
                auc_score = roc_auc_score((all_labels == i).astype(int), all_preds[:, i])
            except ValueError:
                auc_score = float("nan")
            fold_per_class_auc[cls].append(auc_score) 

        print(f"\n=== Fold {fold + 1} Pairwise AUCs (One-vs-One) ===")

        pairwise_auc = {}
        for (i, j) in combinations(range(num_classes), 2):
            cls_i, cls_j = class_names[i], class_names[j]

            # Filter only samples of these two classes
            mask = np.isin(all_labels, [i, j])
            if mask.sum() == 0:
                continue

            y_true_pair = all_labels[mask]
            y_pred_pair = all_preds[mask][:, [i, j]]

            # Compute AUC for distinguishing i vs j
            try:
                auc_pair = roc_auc_score(
                    (y_true_pair == i).astype(int),
                    y_pred_pair[:, 0]  # probability of class i
                )
                pairwise_auc[(cls_i, cls_j)] = auc_pair
            except ValueError:
                pairwise_auc[(cls_i, cls_j)] = float("nan")

            print(f"{cls_i} vs {cls_j}: {pairwise_auc[(cls_i, cls_j)]:.4f}")
    
    print("\n" + "="*60)
    print(f"K-Fold Cross-Validation Results ({k_folds} folds)")
    print("="*60)
    print(f"Mean Validation Accuracy: {np.mean(fold_acc):.4f}")
    print(f"Mean Validation AUC: {np.nanmean(fold_auc):.4f}")
    print(f"AUC Std: {np.nanstd(fold_auc):.4f}")

    return fold_acc, fold_auc, fold_per_class_auc


def grid_search(alpha=0.75): # High alpha to penalize variance more
    param_grid = {
        "second_dim": [64, 128, 256],
        "lr": [1e-4, 1e-3, 1e-2],
        "batch_size": [8, 16, 32],
        "num_epoch": [25, 50, 75],
        "weight_decay": [1e-5, 1e-4, 1e-3],
    }

    grid = list(ParameterGrid(param_grid))
    best_config = None
    best_score = -np.inf
    best_auc = -np.inf
    best_std = np.inf
    all_results = []

    for i, params in enumerate(grid):
        print(f"\n{'='*60}")
        print(f"Grid Search {i+1}/{len(grid)}: {params}")
        print(f"{'='*60}")

        fold_acc, fold_auc, _ = train_kfold(**params)

        mean_auc = np.nanmean(fold_auc)
        std_auc = np.nanstd(fold_auc)
        stability_score = mean_auc - alpha * std_auc  # penalize variance

        print(f"Mean AUC = {mean_auc:.6f}, Std AUC = {std_auc:.6f}, Stability Score = {stability_score:.6f}")

        all_results.append({
            "params": params,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "score": stability_score
        })

        if stability_score > best_score:
            best_score = stability_score
            best_config = params
            best_auc = mean_auc
            best_std = std_auc

    print("\n" + "="*60)
    print("Grid Search Completed")
    print("="*60)
    print(f"Best config: {best_config}")
    print(f"Best stability-aware score: {best_score:.6f}")
    print(f"Corresponding Mean AUC: {best_auc:.6f}, Std AUC: {best_std:.6f}")

    return all_results, best_config

#grid_search()
train_kfold(256, 16, 0.0001, 75, use_smote=True, smote_strategy="auto", smote_k_neighbors=5, weight_decay=1e-4)