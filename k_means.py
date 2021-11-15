"""
Implimentation of k-means clusting to label the iris dataset.
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# get data
X = pd.read_csv('test_data.csv')

# feature scaling
scalar = StandardScaler()
X = scalar.fit_transform(X)

# use elbow method to find max k
error = {}
for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=None,
    ).fit(X)
    error[k] = kmeans.inertia_

# plot the results
plt.plot(list(error.keys()), list(error.values()))
plt.xlabel('K value')
plt.ylabel('Error')
plt.show()

# get preds using optimal k of 3
kmeans_labels = KMeans(
    n_clusters=3,
    n_init=10,
    random_state=None,
).fit_predict(X)
logging.info("Predicted labels: %s", kmeans_labels)

# FINAL RESULTS:
# [1 1 1 1 1 1 1 1 1 1 2 2 0 0 0 0 0 2 0 0 2 0 0 2 0 2 0 0 2 2]
