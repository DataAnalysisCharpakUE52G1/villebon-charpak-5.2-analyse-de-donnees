import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster


class Cluster:
    def __init__(self, features: np.ndarray, **kwargs):
        self.features = features
        self.clust = hac.linkage(self.features, **kwargs)

    def dendogram(self):
        plt.figure(figsize=(25, 10))
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("sample index")
        plt.ylabel("distance")
        hac.dendrogram(self.clust, leaf_rotation=90.0, leaf_font_size=8)
        plt.show()

    def print(self, n: int):
        """
        Print the n clusters
        :param n: number of clusters
        """
        for i, clust in enumerate(self.get(n)):
            print(f"Cluster {i}: {clust}")

    def get(self, n: int) -> np.array:
        """
        Calculate the n clusters
        :param n: number of clusters
        :return clusts: list of clusters
        """
        results = fcluster(self.clust, n, criterion="maxclust")

        s = pd.Series(results)
        clusters = s.unique()
        clusts = [s[s == clust].index for clust in clusters]
        return clusts
