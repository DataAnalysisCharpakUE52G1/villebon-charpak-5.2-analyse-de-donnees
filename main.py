from cluster.cluster import Cluster
import numpy as np
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
import pandas as pd
import datetime as dt
import sys


sys.stdout = open("out", "w")


def from_seconds(sec):
    return (sec // 60) // 60, (sec // 60) % 60, sec % 60


def get_data():
    data = pd.concat([pd.read_csv(link) for link in [f"data/data{i}.csv" for i in range(1, 7)]])
    y = (np.array(data)[:, 0] - 1).astype(bool)
    ids = np.concatenate([[i] * (data.shape[1] - 1) for i in range(data.shape[0])])
    dataframe = pd.DataFrame(
        np.insert(
            np.insert(np.array(data)[:, 1:].reshape(-1, 1), 0, 0, axis=1),
            0,
            ids,
            axis=1,
        )
    )
    dataframe.rename(columns={0: "id", 1: "time", 2: "val"}, inplace=True)
    _format = "%M:%S"
    dataframe["time"] = pd.to_datetime(
        np.array(
            [
                pd.to_datetime(dt.datetime(2000, 1, 1, *from_seconds(i)))
                for i in range(data.shape[1]-1)
            ]*data.shape[0]
        ),
        format=_format,
    )
    dataframe = dataframe.set_index(pd.DatetimeIndex(dataframe["time"]))
    return dataframe, y, data.shape[1]-1


def basic_features_extract(data):
    return extract_features(data, column_id="id", column_sort="time")


def extract_features_from_TS(Data, y):
    extracted_features = basic_features_extract(Data)
    impute(extracted_features)
    # features_filtered = select_features(extracted_features, y)
    features_filtered_direct = extract_relevant_features(
        Data, y, column_id="id", column_sort="time"
    )
    return extracted_features, features_filtered_direct


if __name__ == "__main__":
    X, Y, n = get_data()
    print(X)
    features = np.array(basic_features_extract(X.head(n)))
    np.savetxt("features.csv", features)
