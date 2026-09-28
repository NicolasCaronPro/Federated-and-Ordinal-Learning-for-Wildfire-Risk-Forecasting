# No Consistent Federated Gain in Ordinal Wildfire Risk Prediction

Code accompanying *No Consistent Federated Gain in Ordinal Wildfire Risk Prediction: A Retrospective Analysis of Client Partitions and Server Aggregation* (N. Caron).

## Abstract

Federated learning enables collaborative training without pooling all raw data, but this organizational advantage does not establish an improvement in the quality of an operational risk signal. We investigate this distinction through a retrospective, offline federated simulation on French wildfire data that crosses four experimental client partitions (department, historical similarity, Mediterranean region and season), three federated algorithms (FedAvg, FedALA and MOON), two server aggregation rules (weighted averaging and FLTG) and two backbones (GRU and DilatedCNN). The primary metrics are the monotonic scores k1–k4, which assess whether increasing predicted risk corresponds to increasing observed fire activity over progressively wider ordinal transitions; event recall and the false-positive rate are secondary diagnostics. Across 192 architecture-matched cluster comparisons, 47 of the 48 federated configurations lower k1 in at least one evaluation cluster, and the single configuration that dominates its centralized reference on k1–k4 does so with lower event recall in every cluster. Federation frequently produces supported two-step separation that the centralized references do not display, but event recall decreases in 141 of 192 comparisons. FedALA shows the most favorable balance of the three algorithms, and FedALA with seasonal clients and the recurrent backbone is the only combination that raises recall in every cluster under both server rules, at the cost of more false alarms and of adjacent-level separation in the most active cluster. For the recurrent backbone, the client partition is associated with more of the variation in k1 and k2 than the algorithm or the server rule; for the convolutional backbone, no factor dominates. Five seeds of the retained configuration vary by 6–7% on k1 and k2 and by 1% on recall, much less than the partition-associated differences. Because the factorial comparison uses single runs with unmatched participation and computation, these are nevertheless associations rather than causal effects. They indicate that client organization is an explicit design factor that must be evaluated jointly with the federated algorithm.

## Correspondence with the article

| Article | Module | Main entry points |
|---|---|---|
| 3.2 Department-specific ordinal labels (Eq. 1) | `labels.py` | `DepartmentKMeansLabeler`, `add_ordinal_labels` |
| 4.1 Four experimental client partitions | `partitions.py` | `fit_cluster_encoder`, `assign_clients`, `split_clients` |
| 4.2 Local methods (FedAvg, FedALA Eq. 4, MOON) and server rules (weighted Eq. 3, FLTG) | `federated.py` | `run_federated` (Algorithm 1), `ALAAdapter`, `aggregate_weighted`, `aggregate_fltg` |
| 4.2 WKLOSS (Eq. 6) | `loss_utils.py` | `WKLoss` |
| 4.3 Training protocol, 10-day lookback, zero-class retention search | `training.py` | `make_windows`, `train_local`, `select_retention_fraction` |
| 4.3 Backbones | `architectures.py` | `build_model('GRU' \| 'DilatedCNN', ...)` |
| 5 Monotonic evaluation, k1–k4, transition support, recall, FPR | `evaluation.py` | `MonotonicScorer` (Algorithm 2), `fit_cluster_scorers`, `evaluate_by_cluster` |
| End-to-end pipeline | `example_synthetic.py` | synthetic data, same pipeline |

Two further modules support the pipeline:

- `loss.py` holds other ordinal criteria: all-threshold BCE, foreground Dice, and MCE + WK.
- `tools.py` holds generic helpers: IoU, confidence intervals and serialization.

## Method summary

**Labels.**

- Level 0 means no observed fire.
- For each department, a one-dimensional K-means with four centroids is fitted on the department's positive training counts. The sorted centroids become levels 1–4.
- A department with fewer than four distinct positive values gets fewer positive levels.

**Client partitions.**

| Partition | Clients |
|---|---|
| `department` | one client per department, participating with probability 0.35 per round |
| `cluster-encoder` | four groups from a DTW temporal K-means over the training fire series of the departments |
| `mediterranean` | Mediterranean departments {06, 11, 13, 30, 34, 66, 83, 84} vs the rest |
| `season` | Feb–May, Jun–Sep and Oct–Jan clients, built from the records of all departments |

**Algorithm 1.** Each round runs the following steps.

1. **Retention fraction.** At its first participation, each client selects the fraction of zero-class training samples to keep.
   - The search runs over 0.05–1.00 in steps of 0.05.
   - It maximises the aggregated monotonic score k1+k2+k3+k4 on the client's validation years.
2. **Local training.** The participating clients train with WKLOSS: Adam, lr 5e-4, at most 3,000 epochs per call, local patience 100. The local method sets the starting point:
   - **FedAvg** starts from the global parameters.
   - **FedALA** initialises the adapted higher layers with `u + (w - u) * A`.
     - The element-wise weights `A` are learned and clipped to [0, 1].
     - The first adaptation fits `A` alone; later participations train for one epoch.
   - **MOON** adds a model-contrastive term (µ = 0.5). The global representation is the positive and the previous local representation the negative.
3. **Aggregation.** The server combines the local updates with one of two rules:
   - **Weighted:** weights `n_j / n`, where `n_j` is the training size of client j.
   - **FLTG:** each client update gets a score, its ReLU-clipped cosine with a reference direction. The updates are combined with softmax weights of these scores. The reference depends on whether the server holds trusted data:
     - With trusted server data (`run_federated(..., server_reference=...)`), the reference is a server-side update. Client updates are rescaled to its norm, and updates opposed to it are discarded.
     - Without trusted data, the reference is the mean clipped client update.
4. **Stopping.** Training ends after 60 rounds, or after 10 rounds without improvement of the global validation score. The frozen global model goes to Algorithm 2.

**Algorithm 2.** Within each evaluation cluster:

1. Fit the observed count against a cubic B-spline of the predicted level (`df = 5`).
   - Zone and date fixed effects are included only when they are informative.
   - The model is fitted by OLS with HC1 standard errors.
2. Average the fit over the support to get the adjusted response `mu(s)` of each level.
3. For each transition order k, compute the contrasts of the level pairs listed in `PASSAGES`. Normalise each contrast by the positive reference contrast from the training K-means labels.
4. Compute `SCORE_k = (MED + MIN) / 2 - NEG * (1 + VIOL)` and report it with its pair support. An order without a supported pair scores 0.

## Usage

```bash
pip install -r requirements.txt
python example_synthetic.py --algorithm fedala --server-rule weighted --partition season --backbone GRU
python example_synthetic.py --algorithm moon --server-rule fltg --partition mediterranean --backbone DilatedCNN
```

The BDIFF-derived feature tables are not distributed. To run the pipeline on your own data:

1. Prepare a daily table with one row per department and date. It needs the columns `departement`, `date`, the fire count `nbsinister`, and the features.
2. Reproduce the steps of `example_synthetic.py`:

```python
labeler = DepartmentKMeansLabeler().fit(df[train_mask])
df['level'] = labeler.transform(df)
df['client'] = assign_clients(df, 'season')
clients = {c: (make_windows(df, features, train_mask & (df.client == c)),
               make_windows(df, features, val_mask & (df.client == c))) for c in df.client.unique()}
model, history = run_federated(clients, lambda: build_model('GRU', len(features)),
                               make_windows(df, features, val_mask), FederatedConfig(algorithm='fedala'))
scorers = fit_cluster_scorers(df[train_mask], 'eval_cluster')
```

The article uses this temporal split:

| Split | Years |
|---|---|
| Training | 2017–2020 and 2022 |
| Validation | 2021 and 2024 |
| Test | 2023 |

The defaults of `FederatedConfig` and `LocalTrainingConfig` are the settings of the article. The synthetic example uses a reduced budget instead: 3 rounds, 3 local epochs and a 2-point retention grid. With that budget it runs in a few minutes on a CPU.

## Citation

If you use this code, please cite the article above.
