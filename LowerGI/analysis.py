from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from processing import get_dataset

# Get dataset
X, y, class_names = get_dataset()

def plot_tsne(X, y, class_names, perplexity):
    X_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, learning_rate="auto", max_iter=10000).fit_transform(X)

    plt.figure(figsize=(8,6))
    for i, cls in enumerate(class_names):
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=cls, alpha=0.6)
    plt.title("t-SNE projection of X colored by y perplexity="+str(perplexity))
    plt.legend()
    plt.show()

plot_tsne(X, y, class_names, 3)
plot_tsne(X, y, class_names, 4)
plot_tsne(X, y, class_names, 5)
plot_tsne(X, y, class_names, 6)
plot_tsne(X, y, class_names, 7)
plot_tsne(X, y, class_names, 8)
plot_tsne(X, y, class_names, 9)
plot_tsne(X, y, class_names, 10)
plot_tsne(X, y, class_names, 15)
plot_tsne(X, y, class_names, 20)

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
