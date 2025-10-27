# I plugged in my train.py code into Claude and asked it to incorporate
# bootstrapping over 1000 iterations to get a CI.

import torch
from model import MLPClassifierDeepResidual
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import numpy as np
from sklearn.utils import resample

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


def train_single_fold(X_tr, y_tr, X_val, y_val, second_dim, num_classes, class_weights, device, batch_size=8, lr=1e-3, num_epoch=50):
    """Train a single fold and return validation metrics"""
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    model = MLPClassifierDeepResidual(input_dim=X_tr.shape[1], second_dim=second_dim, num_classes=num_classes).to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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
                               multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    
    return val_acc, auc, all_preds, all_labels


def train_with_bootstrap(second_dim = 128):
    seed = 42
    batch_size = 8
    lr = 1e-3
    num_epoch = 50
    k_folds = 5
    n_bootstrap = 10

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y, class_names = get_dataset()
    num_classes = len(class_names)

    # Split once into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y, shuffle=True
    )

    # Store bootstrap results
    bootstrap_acc = []
    bootstrap_auc = []
    bootstrap_per_class_auc = {cls: [] for cls in class_names}

    print(f"Starting bootstrap with {n_bootstrap} iterations...")
    
    for boot_iter in range(n_bootstrap):
        # Bootstrap resample the training data
        X_boot, y_boot = resample(X_train, y_train, random_state=boot_iter, stratify=y_train)
        
        # Reset torch seed so each model starts with same initialization
        torch.manual_seed(seed)

        # Convert to tensors
        X_boot_tensor = torch.tensor(X_boot.values, dtype=torch.float32)
        y_boot_tensor = torch.tensor(np.array(y_boot), dtype=torch.long)
        
        # Compute class weights for this bootstrap sample
        class_weights = compute_class_weights(y_boot).to(device)
        
        # K-Fold CV on bootstrap sample
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        fold_acc = []
        fold_auc = []
        fold_per_class_auc = {cls: [] for cls in class_names}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_boot_tensor)):
            X_tr, X_val = X_boot_tensor[train_idx], X_boot_tensor[val_idx]
            y_tr, y_val = y_boot_tensor[train_idx], y_boot_tensor[val_idx]
            
            val_acc, auc, all_preds, all_labels = train_single_fold(
                X_tr, y_tr, X_val, y_val, second_dim, num_classes, class_weights, 
                device, batch_size, lr, num_epoch
            )
            
            fold_acc.append(val_acc)
            fold_auc.append(auc)
            
            # Per-class AUC
            for i, cls in enumerate(class_names):
                try:
                    auc_score = roc_auc_score((all_labels == i).astype(int), all_preds[:, i])
                except ValueError:
                    auc_score = float("nan")
                fold_per_class_auc[cls].append(auc_score)
        
        # Average across folds for this bootstrap iteration
        bootstrap_acc.append(np.mean(fold_acc))
        bootstrap_auc.append(np.nanmean(fold_auc))
        
        for cls in class_names:
            bootstrap_per_class_auc[cls].append(np.nanmean(fold_per_class_auc[cls]))
        
        if (boot_iter + 1) % 5 == 0:
            print(f"Completed {boot_iter + 1}/{n_bootstrap} bootstrap iterations")

    # Calculate confidence intervals (95%)
    acc_mean = np.mean(bootstrap_acc)
    acc_ci = np.percentile(bootstrap_acc, [2.5, 97.5])
    
    auc_mean = np.nanmean(bootstrap_auc)
    auc_ci = np.percentile(bootstrap_auc, [2.5, 97.5])
    
    print("\n" + "="*60)
    print(f"BOOTSTRAP RESULTS ({n_bootstrap} iterations)")
    print("="*60)
    print(f"Validation Accuracy: {acc_mean:.4f} (95% CI: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}])")
    print(f"Validation AUC: {auc_mean:.4f} (95% CI: [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}])")
    
    print("\nPer-class AUC with 95% confidence intervals:")
    for cls in class_names:
        cls_auc_mean = np.nanmean(bootstrap_per_class_auc[cls])
        cls_auc_ci = np.percentile(bootstrap_per_class_auc[cls], [2.5, 97.5])
        print(f"{cls}: {cls_auc_mean:.4f} (95% CI: [{cls_auc_ci[0]:.4f}, {cls_auc_ci[1]:.4f}])")
    
    return bootstrap_acc, bootstrap_auc, bootstrap_per_class_auc