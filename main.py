from cluster.cluster import Cluster
import numpy as np

if __name__ == "__main__":
    features = np.array([
        1 * np.linspace(0, 100, 100),
        2 * np.linspace(0, 100, 100),
        np.ones(100),
        1.01 * np.ones(100)
    ])
    clust = Cluster(features)
    clust.print(2)
