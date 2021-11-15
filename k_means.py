"""
Implimentation of k-means clusting to label the iris dataset.
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# get data
X = pd.read_csv('test_data.csv')

# feature scaling
scalar = StandardScaler()
X = scalar.fit_transform(X)

# dim reduction to plot in 2D
pca = PCA(n_components=2)
X = pca.fit_transform(X)

# get preds
kmeans_labels = KMeans(
    n_clusters=3,
    n_init=10,
    random_state=None,
).fit_predict(X)
logging.info("Predicted labels: %s", kmeans_labels)

# plot the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
plt.xlabel('pca first feature')
plt.ylabel('pca second feature')
plt.show()

# FINAL RESULTS:
# [0 0 0 0 0 0 0 0 0 0 1 1 2 2 2 2 2 1 2 2 1 2 2 1 2 1 2 2 1 1]
