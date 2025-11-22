import pandas as pd
#from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_dataset():
    # Load and filter for Lower GI cancers
    processed_dataset = pd.read_csv("TCMA_Genus_Processed/final.csv") 
    lower_gi = processed_dataset[processed_dataset["project"].isin(["READ", "COAD"])]

    # Shared processing
    dataset = lower_gi
    start_idx = dataset.columns.get_loc("HistologicalType") + 1
    X = dataset.iloc[:, start_idx:] 
    X = X.loc[:, (X != 0).any(axis=0)]
    y = dataset["project"].astype("category").cat.codes.values # Encodes alphabetically - ex. COAD is 0 and READ is 1
    class_names = dataset["project"].astype("category").cat.categories

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y, shuffle=True
    )
    feature_names = X_train.columns.tolist()

    print("Class distribution:")
    print(pd.Series(y_train).value_counts())
    print(pd.Series(y_train).value_counts(normalize=True))

    return X, y, class_names, X_train, X_test, y_train, y_test, feature_names


def get_lower_gi_stats():
    dataset = pd.read_csv("TCMA_Genus_Processed/final.csv") 
    lower_gi = dataset[dataset["project"].isin(["READ", "COAD"])]
    return lower_gi["project"].value_counts()

# def apply_smote(X, y, sampling_strategy="auto", k_neighbors=5):
#     sm = SVMSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
#     X_res, y_res = sm.fit_resample(X, y)
#     return X_res, y_res


def apply_pca_95(X_train, X_test, return_scaler=False):
    """
    Apply PCA to retain 95% of variance.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    return_scaler : bool
        Whether to return the scaler object
    
    Returns:
    --------
    X_train_pca, X_test_pca : transformed data
    pca : fitted PCA object
    scaler : fitted StandardScaler (if return_scaler=True)
    """
    # Step 1: Standardize the features (important for PCA)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: Apply PCA with 95% variance retention
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Print information about the reduction
    print(f"\nPCA Results:")
    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Reduced number of components: {pca.n_components_}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Variance per component (first 10): {pca.explained_variance_ratio_[:10]}")
    
    if return_scaler:
        return X_train_pca, X_test_pca, pca, scaler
    return X_train_pca, X_test_pca, pca


# X, y, class_names, X_train, X_test, y_train, y_test, feature_names = get_dataset()
# X_train_pca, X_test_pca, pca = apply_pca_95(X_train=X_train, X_test = X_test)
