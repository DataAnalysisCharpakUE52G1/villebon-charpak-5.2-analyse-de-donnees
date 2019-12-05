from cluster.cluster import Cluster
import numpy as np
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
import pandas as pd
import datetime as dt


def from_seconds(sec):
    return (sec // 60) // 60, (sec // 60) % 60, sec % 60


def get_data(link: str):
    data = pd.read_csv(link)
    inner = np.array(data)[:, 1:].transpose()
    y = (np.array(data)[:, 0] - 1).astype(bool)
    dataframe = pd.DataFrame(
        np.insert(
            np.insert(inner, 0, np.arange(len(inner[:, 0]), dtype=float), axis=1),
            0,
            np.ones(len(inner[:, 0])),
            axis=1,
        )
    )
    dataframe.rename(columns={0: "id", 1: "time"}, inplace=True)
    _format = "%M:%S"
    dataframe["time"] = pd.to_datetime(
        np.array(
            [
                pd.to_datetime(dt.datetime(2000, 1, 1, *from_seconds(i)))
                for i in range(len(inner[:, 0]))
            ]
        ),
        format=_format,
    )
    dataframe = dataframe.set_index(pd.DatetimeIndex(dataframe["time"]))
    return dataframe, y


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
    X, Y = get_data("data/data1.csv")
    X = X[["id", "time", *range(2, 20)]]
    print(X.head())
    features = basic_features_extract(X)
    print(features)
    clust = Cluster(features)
    clust.print(2)
