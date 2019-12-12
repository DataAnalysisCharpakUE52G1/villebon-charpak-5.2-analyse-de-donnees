from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from math import sqrt
from fastdtw import fastdtw


def print(msg):
    with open("out.log", "a+") as f:
        f.write(msg)


def dtw(s1, s2):
    return fastdtw(s1, s2)[0]


def norm(tab):
    ma = np.array(
        [
            max(abs(max(e)), abs(min(e))) if max(abs(max(e)), abs(min(e))) != 0 else 1
            for e in tab.transpose()
        ]
    )
    return tab / ma


n = 100


ts = norm(
    np.concatenate(
        list(
            (
                np.array(pd.read_csv(f"data/data{i}.csv"))[:, 1:]
                for i in range(1, 7)
            )
        ),
        axis=0,
    )
)


print(ts.shape)
to_hist = []
m = len(ts)

for i in range(0, m - 1):
    for j in range(i + 1, m):
        to_hist.append(dtw(ts[i], ts[j]))

to_hist = np.array(to_hist)

print(np.median(to_hist))
print(to_hist.std())

for eps in np.arange(0, to_hist.std()*2, 0.1):
    db = DBSCAN(eps=eps).fit(ts)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_


    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(f"Estimated number of clusters: {n_clusters_}")
    print(f"Estimated number of noise points: {n_noise_}")


"""
clusters = [np.arange(ft.shape[0])[labels == i] for i in set(list(labels)) - {-1}]
data = norm(
    np.concatenate(
        list(
            (
                np.array(pd.read_csv(f"data/data{i}.csv"))[:, 1:]
                for i in range(1, 7)
            )
        ),
        axis=0,
    )
)


x = np.arange(data.shape[1])
for i, cluster in enumerate(clusters):
    print(f"Generating cluster {i}: {len(cluster)} graphs")
    graph = sns.JointGrid(x, x)
    prec = 0
    end = len(cluster)
    for j, y in enumerate(data[list(cluster)]):
        if int(100*j/end) != prec:
            prec = int(100*j/end)
            if prec < 10:
                print(f"0{prec}%")
            else:
                print(f"{prec}%")
        graph.y = y
        graph.plot_joint(plt.scatter, marker="+")
        graph.plot_marginals(sns.distplot)
    plt.title(f"Cluster {i}")
    plt.show()
"""
