import pandas as pd
from imblearn.over_sampling import SVMSMOTE

def get_dataset():
    # Load and filter for Lower GI cancers
    processed_dataset = pd.read_csv("TCMA_Genus_Processed/final.csv") 
    lower_gi = processed_dataset[processed_dataset["project"].isin(["READ", "COAD"])]

    # Shared processing
    dataset = lower_gi
    start_idx = dataset.columns.get_loc("HistologicalType") + 1
    X = dataset.iloc[:, start_idx:] 
    X = X.loc[:, (X != 0).any(axis=0)]
    y = dataset["project"].astype("category").cat.codes.values
    class_names = dataset["project"].astype("category").cat.categories
    return X, y, class_names


def get_lower_gi_stats():
    dataset = pd.read_csv("TCMA_Genus_Processed/final.csv") 
    lower_gi = dataset[dataset["project"].isin(["READ", "COAD"])]
    return lower_gi["project"].value_counts()

def apply_smote(X, y, sampling_strategy="auto", k_neighbors=5):
    sm = SVMSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res