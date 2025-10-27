import torch
from model import MLPClassifierDeepResidual
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import classification_report

def get_dataset():
    dataset = pd.read_csv("TCMA_Genus_Processed/final.csv")
    start_idx = dataset.columns.get_loc("HistologicalType") + 1
    X = dataset.iloc[:, start_idx:] 
    y = dataset["project"].astype("category").cat.codes.values # Encode string labels
    class_names = dataset["project"].astype("category").cat.categories
    return X, y, class_names


# I got help from chat gpt in weighting to balance classees
def compute_class_weights(y):
    """Compute inverse frequency class weights"""
    classes, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)  # normalize to num classes
    class_weights = torch.tensor(weights, dtype=torch.float32)
    return class_weights


# Structure of train from Deep Learning course with modification to fit this project
def train():
    seed = 42
    batch_size = 8
    lr = 1e-3
    num_epoch = 50
    k_folds = 5


    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)   


    X, y, class_names = get_dataset()
    num_classes = len(class_names)

    class_weights = compute_class_weights(y).to(device)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y, shuffle=True)

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)

    kf =  KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_results_acc = []
    fold_results_auc = []
    per_class_auc_folds = {cls: [] for cls in class_names}


    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)): # ChatGPT helped me set up KFold Cross Validation

        # The next five lines are from ChatGPT
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
       
        model = MLPClassifierDeepResidual(input_dim=X.shape[1], num_classes=num_classes).to(device)
        
        #loss_func = ClassificationLoss()
        loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) # set LR; otherwise use default parameters for Adam optimizer

   
        metrics = {"train_acc": [], "val_acc": [], "val_auc": []}

        for epoch in range(num_epoch):
            # clear metrics at beginning of epoch
            for key in metrics:
                metrics[key].clear()

            model.train()

            for x_data, y_data in train_loader:
                x_data, y_data = x_data.to(device), y_data.to(device)
                pred = model(x_data)


                # calculate training accuracy
                acc = (pred.argmax(dim=1) == y_data).float().mean() # used copilot
                metrics["train_acc"].append(acc)

                loss_val = loss_func(pred, y_data)
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

            # disable gradient computation and switch to evaluation mode\

            model.eval()
            all_preds, all_labels = [], []
            with torch.inference_mode():
                for x_data, y_data in val_loader:
                    x_data, y_data = x_data.to(device), y_data.to(device)

                    pred = model(x_data)
                    acc = (pred.argmax(dim=1) == y_data).float().mean()
                    metrics["val_acc"].append(acc)

                    # ChatGPT helped me here with AUC
                    probs = F.softmax(pred, dim=1).cpu().numpy() 
                    labels = y_data.cpu().numpy()
                    all_preds.append(probs)
                    all_labels.append(labels)


            # Chat GPT helped me here to calculate AUC
            # Convert lists to numpy array
            # concatenate all batches
            all_preds = np.concatenate(all_preds, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            try:
                if num_classes == 2:
                    # for binary, use probs of positive class
                    auc = roc_auc_score(all_labels, all_preds[:, 1])
                else:
                    # for multiclass
                    auc = roc_auc_score(
                        y_true=all_labels,
                        y_score=all_preds,
                        multi_class="ovr",
                        average="macro"
                    )
            except ValueError:
                auc = float("nan")  # handle rare cases if a class is missing in this fold

            metrics["val_auc"].append(auc)

            epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
            epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
            epoch_val_auc = metrics["val_auc"][-1]

            # print on first, last, every 10th epoch
            if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
                print(
                    f"Fold {fold + 1:2d} - "
                    f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                    f"train_acc={epoch_train_acc:.4f} "
                    f"val_acc={epoch_val_acc:.4f}",
                    f"val_auc={epoch_val_auc:.4f}"
                )

        # Chat GPT helped me here with the per-class AUC
        for i, cls in enumerate(class_names):
            try:
                auc_score = roc_auc_score((all_labels == i).astype(int), all_preds[:, i])
            except ValueError:
                auc_score = float("nan")
            per_class_auc_folds[cls].append(auc_score)
                
        fold_results_acc.append(epoch_val_acc.item())
        fold_results_auc.append(epoch_val_auc)
            
    avg_val_acc = torch.as_tensor(fold_results_acc).mean()
    print(f"Average validation accuracy across {k_folds} folds: {avg_val_acc:.4f}")

    avg_val_auc = torch.as_tensor(fold_results_auc).mean()
    print(f"Average validation AUC across {k_folds} folds: {avg_val_auc:.4f}")

    print("\nAverage per-class AUC across folds:")
    for cls in class_names:
        mean_auc = np.nanmean(per_class_auc_folds[cls])
        print(f"{cls}: {mean_auc:.4f}")

