"""Department-specific ordinal risk labels (Section 3.2, Eq. 1).

Level 0 is an observed zero fire count. For each department, four one-dimensional
K-means centroids are fitted on its positive training counts; they are sorted in
ascending order and remapped to levels 1-4, and every positive count is assigned to
its nearest centroid. Centroids are fitted on training data only and held fixed when
labelling validation and test data.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class DepartmentKMeansLabeler:
    """Per-department K-means discretisation of a daily count into levels 0-4."""

    def __init__(self, n_positive_levels=4, random_state=42):
        self.n_positive_levels = n_positive_levels
        self.random_state = random_state
        self.models_ = {}      # zone -> fitted KMeans (None if no positive training count)
        self.label_maps_ = {}  # zone -> {kmeans label: ordinal level}

    def fit(self, df, zone_col='departement', target_col='nbsinister'):
        """Fit the centroids of each department on its positive training counts."""
        for zone, sub in df.groupby(zone_col):
            positives = sub[target_col].to_numpy(dtype=float)
            positives = positives[positives > 0].reshape(-1, 1)
            n_unique = np.unique(positives).shape[0]
            if n_unique == 0:
                # No positive training count: every record of this department is level 0.
                self.models_[zone] = None
                continue

            # A department with fewer distinct positive values cannot support four centroids.
            n_clusters = min(self.n_positive_levels, n_unique)
            model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            model.fit(positives)

            # Ordering comes from the scalar centroids, not from the arbitrary K-means labels.
            order = np.argsort(model.cluster_centers_.flatten())
            self.models_[zone] = model
            self.label_maps_[zone] = {label: level + 1 for level, label in enumerate(order)}
        return self

    def transform(self, df, zone_col='departement', target_col='nbsinister'):
        """Return the ordinal level (0-4) of every row of ``df``."""
        levels = np.zeros(len(df), dtype=int)
        counts = df[target_col].to_numpy(dtype=float)
        zones = df[zone_col].to_numpy()
        for zone in np.unique(zones):
            model = self.models_.get(zone)
            if model is None:
                continue
            mask = (zones == zone) & (counts > 0)
            if not mask.any():
                continue
            kmeans_labels = model.predict(counts[mask].reshape(-1, 1))
            levels[mask] = [self.label_maps_[zone][label] for label in kmeans_labels]
        return levels

    def fit_transform(self, df, zone_col='departement', target_col='nbsinister'):
        return self.fit(df, zone_col, target_col).transform(df, zone_col, target_col)


def add_ordinal_labels(df_train, dfs, zone_col='departement', target_col='nbsinister',
                       label_col='level'):
    """Fit the labeler on ``df_train`` and add ``label_col`` to every DataFrame of ``dfs``."""
    labeler = DepartmentKMeansLabeler().fit(df_train, zone_col, target_col)
    out = []
    for d in dfs:
        d = d.copy()
        d[label_col] = labeler.transform(d, zone_col, target_col)
        out.append(d)
    return labeler, out
