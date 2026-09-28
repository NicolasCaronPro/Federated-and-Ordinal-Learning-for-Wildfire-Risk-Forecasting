"""Local training (Section 4.3).

- 10-day lookback windows: the current day plus the 10 previous days of each department.
- WKLOSS (quadratic weighted kappa) optimised with Adam; early stopping on the validation
  loss with a local patience, inside a per-call epoch ceiling.
- MOON adds a model-contrastive term (Li et al., 2021) when a previous local model exists.
- At a client's first participation, the retained fraction of zero-class training samples is
  selected over 0.05-1.00 (step 0.05) by maximising the aggregated monotonic score k1+k2+k3+k4
  on the client's validation years.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluation import MonotonicScorer, aggregated_monotonic_score
from loss_utils import WKLoss

RETENTION_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)


# ─────────────────────────────── Data ──────────────────────────────────────
@dataclass
class WindowSet:
    """Lookback windows of one split: inputs X (N, features, lookback + 1) and their metadata."""
    X: np.ndarray
    level: np.ndarray   # ordinal training label L in {0,...,4}
    count: np.ndarray   # observed fire count Y (evaluation response)
    zone: np.ndarray
    date: np.ndarray

    def __len__(self):
        return len(self.level)

    def subset(self, idx):
        return WindowSet(self.X[idx], self.level[idx], self.count[idx], self.zone[idx], self.date[idx])

    def loader(self, batch_size=64, shuffle=False):
        ds = TensorDataset(torch.as_tensor(self.X, dtype=torch.float32), torch.as_tensor(self.level, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def make_windows(df, feature_cols, rows, lookback=10, label_col='level', target_col='nbsinister',
                 zone_col='departement', date_col='date'):
    """Build the windows of the rows selected by the boolean mask ``rows``.

    ``df`` must contain the complete daily series of each department, so that the first
    validation or test days can use their preceding days as history. Rows without a full
    lookback history are dropped.
    """
    X, level, count, zone, date = [], [], [], [], []
    for z, sub in df.assign(_row=np.asarray(rows)).sort_values(date_col).groupby(zone_col, sort=False):
        feats = sub[feature_cols].to_numpy(dtype=np.float32)
        keep = np.flatnonzero(sub['_row'].to_numpy())
        keep = keep[keep >= lookback]
        for i in keep:
            X.append(feats[i - lookback:i + 1].T)   # (features, lookback + 1)
        level.append(sub[label_col].to_numpy()[keep])
        count.append(sub[target_col].to_numpy()[keep])
        zone.append(np.full(len(keep), z))
        date.append(sub[date_col].to_numpy()[keep])
    if not X:
        raise ValueError('No window could be built: check the selected rows and the lookback.')
    return WindowSet(np.stack(X), np.concatenate(level).astype(int), np.concatenate(count).astype(float),
                     np.concatenate(zone), np.concatenate(date))


# ─────────────────────────────── Losses ────────────────────────────────────
def moon_contrastive_loss(z, z_prev, z_glob, temperature):
    """MOON term: pull the local representation towards the global one, away from the previous local one."""
    z, z_prev, z_glob = F.normalize(z, dim=1), F.normalize(z_prev, dim=1), F.normalize(z_glob, dim=1)
    logits = torch.stack([torch.sum(z * z_glob, dim=1), torch.sum(z * z_prev, dim=1)], dim=1) / temperature
    return F.cross_entropy(logits, torch.zeros(z.size(0), dtype=torch.long, device=z.device))


# ─────────────────────────────── Training ──────────────────────────────────
@dataclass
class LocalTrainingConfig:
    lr: float = 5e-4
    batch_size: int = 64
    max_epochs: int = 3000      # per-call ceiling inside a communication round
    patience: int = 100         # local early-stopping patience on the validation loss
    num_classes: int = 5
    device: str = 'cpu'


def predict_levels(model, windows, batch_size=256, device='cpu'):
    """Predicted risk level S = argmax of the softmax output."""
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in windows.loader(batch_size):
            out, _, _ = model(x.to(device))
            preds.append(out.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def train_local(model, train, val, cfg, max_epochs=None, contrastive=None, ala=None, update_params=True):
    """Train ``model`` in place with WKLOSS and restore its best validation state.

    contrastive : optional dict with ``prev_model``, ``global_model``, ``temperature``, ``mu`` (MOON).
    ala         : optional FedALA adapter whose ``step()`` is called after each backward pass.
    update_params : False for FedALA's weight-only fitting call (the optimizer does not step).
    """
    device = cfg.device
    model.to(device)
    criterion = WKLoss(cfg.num_classes, penalization_type='quadratic', use_logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    max_epochs = cfg.max_epochs if max_epochs is None else max_epochs
    if contrastive is not None:
        for frozen in (contrastive['prev_model'], contrastive['global_model']):
            frozen.to(device).eval()

    best_loss, best_state, wait = float('inf'), deepcopy(model.state_dict()), 0
    for epoch in range(max_epochs):
        model.train()
        for x, y in train.loader(cfg.batch_size, shuffle=True):
            if x.shape[0] < 2:   # BatchNorm needs more than one sample
                continue
            x, y = x.to(device), y.to(device)
            _, logits, hidden = model(x)
            loss = criterion(logits, y)
            if contrastive is not None:
                with torch.no_grad():
                    z_prev = contrastive['prev_model'](x)[2]
                    z_glob = contrastive['global_model'](x)[2]
                loss = loss + contrastive['mu'] * moon_contrastive_loss(hidden, z_prev, z_glob, contrastive['temperature'])
            optimizer.zero_grad()
            loss.backward()
            if ala is not None:
                ala.step()
            if update_params:
                optimizer.step()

        val_loss = evaluate_loss(model, val, criterion, cfg)
        if val_loss < best_loss:
            best_loss, best_state, wait = val_loss, deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if update_params:
        model.load_state_dict(best_state)
    return best_loss


def evaluate_loss(model, windows, criterion, cfg):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in windows.loader(cfg.batch_size):
            _, logits, _ = model(x.to(cfg.device))
            total += float(criterion(logits, y.to(cfg.device))) * len(y)
            n += len(y)
    return total / max(n, 1)


def subsample_zeros(train, fraction, rng):
    """Keep every fire-positive window and a fraction of the zero-class windows."""
    zeros = np.flatnonzero(train.level == 0)
    kept = rng.choice(zeros, size=int(round(fraction * len(zeros))), replace=False) if len(zeros) else zeros
    return train.subset(np.sort(np.concatenate([np.flatnonzero(train.level > 0), kept])))


def select_retention_fraction(initial_model, train, val, cfg, grid=RETENTION_GRID, seed=42, contrastive=None):
    """Select the zero-class retention fraction of a client (first participation).

    A model is trained from ``initial_model`` for every fraction of ``grid`` and scored on
    the client's validation windows; the scorer's reference contrasts come from the client's
    training K-means labels. Returns ``(fraction, trained_model, training_subset, scores)``.
    """
    scorer = MonotonicScorer().fit_reference(train.level, train.count, train.zone, train.date)
    rng = np.random.default_rng(seed)
    best = None
    history = {}
    for fraction in grid:
        subset = subsample_zeros(train, fraction, rng)
        model = deepcopy(initial_model)
        train_local(model, subset, val, cfg, contrastive=contrastive)
        agg = aggregated_monotonic_score(scorer.score(predict_levels(model, val, device=cfg.device),
                                                      val.count, val.zone, val.date))
        history[float(fraction)] = agg
        if best is None or agg > best[0]:
            best = (agg, float(fraction), model, subset)
    return best[1], best[2], best[3], history
