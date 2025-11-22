import pandas as pd
from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import train_test_split

def get_dataset():
    # Load and filter for Upper GI cancers
    processed_dataset = pd.read_csv("TCMA_Genus_Processed/final.csv") 
    upper_gi = processed_dataset[processed_dataset["project"].isin(["ESCA", "STAD"])]

    # Shared processing
    dataset = upper_gi
    start_idx = dataset.columns.get_loc("HistologicalType") + 1
    X = dataset.iloc[:, start_idx:] 
    X = X.loc[:, (X != 0).any(axis=0)]
    y = dataset["project"].astype("category").cat.codes.values # Encodes alphabetically 
    class_names = dataset["project"].astype("category").cat.categories

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y, shuffle=True
    )
    feature_names = X_train.columns.tolist()

    print("Class distribution:")
    print(pd.Series(y_train).value_counts())
    print(pd.Series(y_train).value_counts(normalize=True))

    return X, y, class_names, X_train, X_test, y_train, y_test, feature_names


def get_upper_gi_stats():
    dataset = pd.read_csv("TCMA_Genus_Processed/final.csv") 
    upper_gi = dataset[dataset["project"].isin(["ESCA", "STAD"])]
    return upper_gi["project"].value_counts()

def apply_smote(X, y, sampling_strategy="auto", k_neighbors=5):
    sm = SVMSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res