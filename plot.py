import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    end = 1000
    ind = (0, 5)
    #ind = (0, 1, 3, 5, 6, 8)
    data = np.array(pd.read_csv("data/data1.csv").head(10))[ind, 1:end+1]
    print(data)
    print(data.shape)
    x = np.arange(end)
    graph = sns.JointGrid(x, x)
    for i, y in enumerate(data[:len(ind)]):
        graph.y = y
        graph.plot_joint(plt.plot)
        graph.plot_marginals(sns.distplot)
    plt.show()
