from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from train_hyperparameter_optim import get_dataset

X, y, class_names = get_dataset()

X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

plt.figure(figsize=(8,6))
for i, cls in enumerate(class_names):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=cls, alpha=0.6)
plt.title("t-SNE projection of X colored by y")
plt.legend()
plt.show()