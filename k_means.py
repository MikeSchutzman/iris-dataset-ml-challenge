"""
Implimentation of k-means clusting to label the iris dataset.
"""
import logging
import pandas as pd
from sklearn.cluster import KMeans


if __name__ == "__main__":
    X = pd.read_csv('test_data_scaled.csv')
    kmeans_labels = KMeans(
        n_clusters=3,
        n_init=10,
        random_state=None,
    ).fit_predict(X)
    logging.info("Predicted labels: %s", kmeans_labels)

# FINAL RESULTS:
# [0 0 0 0 0 0 0 0 0 0 1 1 2 2 2 2 2 2 2 2 1 1 2 1 1 1 2 2 1 1]
