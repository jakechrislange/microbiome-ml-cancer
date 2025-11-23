from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from processing import get_dataset

# Get dataset
X, y, class_names, X_train, X_test, y_train, y_test, feature_names = get_dataset()

def plot_tsne(X, y, class_names, perplexity):
    X_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, learning_rate="auto", max_iter=10000).fit_transform(X)

    plt.figure(figsize=(8,6))
    for i, cls in enumerate(class_names):
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=cls, alpha=0.6)
    plt.title("t-SNE projection of X colored by y perplexity="+str(perplexity))
    plt.legend()
    plt.show()

plot_tsne(X, y, class_names, 5)
plot_tsne(X, y, class_names, 10)
plot_tsne(X, y, class_names, 15)
plot_tsne(X, y, class_names, 20)
plot_tsne(X, y, class_names, 25)
plot_tsne(X, y, class_names, 30)
plot_tsne(X, y, class_names, 35)
plot_tsne(X, y, class_names, 40)
plot_tsne(X, y, class_names, 45)
# plot_tsne(X, y, class_names, 7)
# plot_tsne(X, y, class_names, 8)
# plot_tsne(X, y, class_names, 9)
# plot_tsne(X, y, class_names, 10)
# plot_tsne(X, y, class_names, 15)
# plot_tsne(X, y, class_names, 20)

# plot_tsne(X, y, class_names, 25)
# plot_tsne(X, y, class_names, 30)
# plot_tsne(X, y, class_names, 35)
# plot_tsne(X, y, class_names, 40)
# plot_tsne(X, y, class_names, 45)
# plot_tsne(X, y, class_names, 50)
# plot_tsne(X, y, class_names, 55)
# plot_tsne(X, y, class_names, 60)
# plot_tsne(X, y, class_names, 65)
# plot_tsne(X, y, class_names, 70)
# plot_tsne(X, y, class_names, 75)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def estimate_bayes_error_knn(X, y, k=1):
    loo = LeaveOneOut()
    errors = []
    knn = KNeighborsClassifier(n_neighbors=k)

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        errors.append(int(y_pred[0] != y_test[0]))

    loo_error = np.mean(errors)
    return loo_error

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

bayes_error_estimate = estimate_bayes_error_knn(X_train_scaled, y_train, k=1)
print(f"Estimated Bayes Error Rate (kNN LOO error): {bayes_error_estimate:.4f}") # may need to divide by 2?

# Estimated Bayes Error Rate (kNN LOO error): 0.3611
#Your model’s achievable performance is fundamentally limited by these data characteristics, not just by model design or tuning.