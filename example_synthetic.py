"""End-to-end example on synthetic data (the BDIFF dataset is not distributed with this repository).

It runs the full pipeline of the paper on a small, fast configuration:
labels (Section 3.2) -> client partition (4.1) -> federated training (4.2, Algorithm 1)
-> monotonic evaluation by evaluation cluster (5, Algorithm 2).

    python example_synthetic.py --algorithm fedala --server-rule weighted --partition season
"""

import argparse

import numpy as np
import pandas as pd

from architectures import build_model
from evaluation import evaluate_by_cluster, fit_cluster_scorers
from federated import FederatedConfig, run_federated
from labels import DepartmentKMeansLabeler
from partitions import assign_clients, fit_cluster_encoder
from training import LocalTrainingConfig, make_windows, predict_levels

FEATURES = ['temperature', 'humidity', 'wind', 'drought']


def synthetic_data(n_departments=8, start='2017-01-01', end='2023-12-31', seed=0):
    """Daily department series whose fire counts increase with a seasonal drought index."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq='D')
    departments = [6, 13, 83, 1, 25, 69, 78, 33][:n_departments]   # Mediterranean and other departments
    rows = []
    for i, dept in enumerate(departments):
        season = np.sin(2 * np.pi * (dates.dayofyear - 100) / 365.25)
        temperature = 15 + 10 * season + rng.normal(0, 2, len(dates))
        humidity = 60 - 20 * season + rng.normal(0, 5, len(dates))
        wind = rng.gamma(2.0, 2.0, len(dates))
        drought = np.clip(season + 0.3 * (dept in (6, 13, 83)) + rng.normal(0, 0.3, len(dates)), -2, 3)
        rate = np.exp(-2.5 + 1.2 * drought + 0.05 * wind + 0.1 * i)
        rows.append(pd.DataFrame({'departement': dept, 'date': dates, 'temperature': temperature,
                                  'humidity': humidity, 'wind': wind, 'drought': drought,
                                  'nbsinister': rng.poisson(rate)}))
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='fedavg', choices=['fedavg', 'fedala', 'moon'])
    parser.add_argument('--server-rule', default='weighted', choices=['weighted', 'fltg'])
    parser.add_argument('--partition', default='season', choices=['department', 'cluster-encoder', 'mediterranean', 'season'])
    parser.add_argument('--backbone', default='GRU', choices=['GRU', 'DilatedCNN'])
    args = parser.parse_args()

    df = synthetic_data()
    year = df['date'].dt.year
    split = np.where(year.isin([2021, 2024]), 'val', np.where(year == 2023, 'test', 'train'))   # Section 3.3
    df[FEATURES] = (df[FEATURES] - df.loc[split == 'train', FEATURES].mean()) / df.loc[split == 'train', FEATURES].std()

    # Department-specific ordinal labels fitted on the training years (Eq. 1).
    labeler = DepartmentKMeansLabeler().fit(df[split == 'train'])
    df['level'] = labeler.transform(df)

    # Training clients and evaluation clusters (here: the temporal K-means groups).
    cluster_map = fit_cluster_encoder(df[split == 'train'], n_clusters=2)
    df['client'] = assign_clients(df, args.partition, cluster_map=cluster_map)
    df['eval_cluster'] = df['departement'].map(cluster_map)

    windows = {s: make_windows(df, FEATURES, split == s) for s in ('train', 'val', 'test')}
    # Each client keeps the full daily series for the history, but only its own records as targets.
    clients = {c: tuple(make_windows(df, FEATURES, (split == s) & (df['client'] == c).to_numpy()) for s in ('train', 'val'))
               for c in sorted(df['client'].dropna().unique())}

    cfg = FederatedConfig(algorithm=args.algorithm, server_rule=args.server_rule, max_rounds=3, global_patience=2,
                          participation=0.35 if args.partition == 'department' else None,
                          retention_grid=np.array([0.25, 1.0]),
                          local=LocalTrainingConfig(max_epochs=3, patience=2, batch_size=128))
    model, history = run_federated(clients, lambda: build_model(args.backbone, len(FEATURES)), windows['val'], cfg)
    print('global validation score per round:', [round(float(s), 3) for s in history['global_score']])

    # Algorithm 2 on the test year, one scorer per evaluation cluster fitted on the training years.
    train_df = df[split == 'train']
    scorers = fit_cluster_scorers(train_df, 'eval_cluster')
    test = windows['test']
    pred = pd.DataFrame({'departement': test.zone, 'date': test.date, 'nbsinister': test.count,
                         'prediction': predict_levels(model, test)})
    pred['eval_cluster'] = pred['departement'].map(cluster_map)
    print(evaluate_by_cluster(pred, 'prediction', scorers, 'eval_cluster')
          [['cluster', 'k1', 'k2', 'k3', 'k4', 'support_k1', 'support_k2', 'recall', 'fpr']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
