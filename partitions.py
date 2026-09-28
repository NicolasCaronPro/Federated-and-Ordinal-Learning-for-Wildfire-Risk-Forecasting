"""Experimental client partitions (Section 4.1, Table 1).

A partition assigns every record to a training client. The evaluation clusters C0-C3
are independent of the partition used for training.

- ``department``      one client per department (participation 0.35 per round, see federated.py)
- ``cluster-encoder`` four groups from a temporal K-means on departmental fire-activity histories
- ``mediterranean``   Mediterranean vs non-Mediterranean departments
- ``season``          high (June-September), intermediate (February-May), low (October-January)
"""

import warnings

import numpy as np
import pandas as pd

PARTITIONS = ('department', 'cluster-encoder', 'mediterranean', 'season')

# Mainland Mediterranean departments (Corsica is excluded from the study area).
MEDITERRANEAN_DEPARTMENTS = {6, 11, 13, 30, 34, 66, 83, 84}

SEASON_OF_MONTH = {
    2: 'medium', 3: 'medium', 4: 'medium', 5: 'medium',
    6: 'high', 7: 'high', 8: 'high', 9: 'high',
    10: 'low', 11: 'low', 12: 'low', 1: 'low',
}


def fit_cluster_encoder(df_train, zone_col='departement', date_col='date', target_col='nbsinister',
                        n_clusters=4, random_state=42):
    """Temporal K-means (DTW) on the daily fire-count series of each department.

    Only the training years are used, so no validation or test outcome enters the
    client definition. Returns a mapping ``department -> cluster index``.
    """
    series = (df_train.pivot_table(index=zone_col, columns=date_col, values=target_col, aggfunc='sum')
              .sort_index(axis=1).fillna(0.0))
    X = series.to_numpy(dtype=float)
    try:
        from tslearn.clustering import TimeSeriesKMeans
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw', max_iter=10, random_state=random_state)
        labels = model.fit_predict(X[:, :, None])
    except ImportError:
        warnings.warn('tslearn is not installed: falling back to Euclidean K-means on the fire-count series.')
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(X)
    return dict(zip(series.index, labels))


def assign_clients(df, partition, zone_col='departement', date_col='date', cluster_map=None):
    """Return the training client of every row of ``df`` for the given partition."""
    if partition == 'department':
        return df[zone_col].astype(str)
    if partition == 'cluster-encoder':
        if cluster_map is None:
            raise ValueError("The 'cluster-encoder' partition needs the mapping from fit_cluster_encoder().")
        return df[zone_col].map(cluster_map).astype(str)
    if partition == 'mediterranean':
        return df[zone_col].isin(MEDITERRANEAN_DEPARTMENTS).map({True: 'mediterranean', False: 'other'})
    if partition == 'season':
        return pd.to_datetime(df[date_col]).dt.month.map(SEASON_OF_MONTH)
    raise ValueError(f'Unknown partition {partition!r}; expected one of {PARTITIONS}.')


def split_clients(df_train, df_val, partition, zone_col='departement', date_col='date', cluster_map=None):
    """Split the training and validation records into ``{client: (train, val)}``.

    Clients with an empty training or validation set are dropped.
    """
    c_train = assign_clients(df_train, partition, zone_col, date_col, cluster_map)
    c_val = assign_clients(df_val, partition, zone_col, date_col, cluster_map)
    clients = {}
    for client in sorted(set(c_train.dropna()) & set(c_val.dropna())):
        tr, va = df_train[c_train == client], df_val[c_val == client]
        if len(tr) and len(va):
            clients[client] = (tr, va)
    return clients
