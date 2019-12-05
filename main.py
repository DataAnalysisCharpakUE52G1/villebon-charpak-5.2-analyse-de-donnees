from cluster.cluster import Cluster
import numpy as np
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features



def Extract_features_from_TS(Data, y):
    extracted_features = extract_features(Data, column_id="id", column_sort="time")
    impute(extracted_features)
    features_filtered = select_features(extracted_features, y)
    features_filtered_direct = extract_relevant_features(timeseries, y,
                                                     column_id='id', column_sort='time')
    return extracted_features, features_filtered_direct

    
if __name__ == "__main__":
    features = np.array([
        1 * np.linspace(0, 100, 100),
        2 * np.linspace(0, 100, 100),
        np.ones(100),
        1.01 * np.ones(100)
    ])
    clust = Cluster(features)
    clust.print(2)
