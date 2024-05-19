import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, cluster
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

# Завантаження даних Iris
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# Ініціалізація та навчання KMeans
cluster.KMeans(n_clusters=8, init='k-means + +', n_init=10, max_iter=300, tol=0.0001,
               verbose=0, random_state=None, copy_x=True, algorithm='auto')
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

# Прогнозування кластерів
y_kmeans = kmeans.predict(X)

# Візуалізація результатів
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

# Функція для пошуку кластерів
def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        labels = pairwise_distances_argmin(X, centers)
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels

# Використання функції для пошуку кластерів
centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
