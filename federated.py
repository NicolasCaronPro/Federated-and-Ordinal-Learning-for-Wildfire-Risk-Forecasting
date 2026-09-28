"""Client-server federated training (Section 4.2, Algorithm 1).

Local methods
    fedavg  the client starts from the broadcast global parameters.
    fedala  adaptive local aggregation (Zhang et al., 2023): the adapted higher layers are
            initialised with u + (w - u) * A (Eq. 4), where the element-wise weights A are learned
            locally and clipped to [0, 1]; the lower layers are replaced by the global parameters.
    moon    model-contrastive training (Li et al., 2021): the frozen global representation is the
            positive reference and the previous local representation the negative one.

Server rules
    weighted  FedAvg weighted by the client sample counts n_j (Eq. 3).
    fltg      direction-filtered aggregation of the client updates (Wen et al., 2025): each update
              is scored by its ReLU-clipped cosine with a reference direction and the retained
              updates are combined with softmax weights. With trusted server data
              (``run_federated(..., server_reference=...)``) the reference is a server-side update
              and the client updates are rescaled to its norm (magnitude normalisation); without
              it, the reference is the mean clipped client update, as in the reported runs.

Every client selects its zero-class retention fraction at its first participation (training.py).
Department clients participate with probability 0.35 per round; other partitions use all clients.
"""

import random
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch

from tools import iou_score
from training import (LocalTrainingConfig, RETENTION_GRID, predict_levels, select_retention_fraction,
                      train_local)


@dataclass
class FederatedConfig:
    algorithm: str = 'fedavg'            # fedavg | fedala | moon
    server_rule: str = 'weighted'        # weighted | fltg
    max_rounds: int = 60
    global_patience: int = 10
    participation: float = None          # None: every client; 0.35 for the department partition
    retention_grid: np.ndarray = field(default_factory=lambda: RETENTION_GRID)
    local: LocalTrainingConfig = field(default_factory=LocalTrainingConfig)
    # FedALA
    ala_layers: tuple = ('linear2',)     # adapted higher layers; the other layers take the global values
    ala_eta: float = 0.1
    ala_init: float = 0.5
    # MOON
    moon_temperature: float = 1.0
    moon_mu: float = 0.5
    # FLTG
    fltg_clip_tau: float = 100.0
    fltg_temperature: float = 1.0
    fltg_min_clients: int = 2
    seed: int = 42


# ─────────────────────────────── Server rules ──────────────────────────────
def aggregate_weighted(local_states, sample_counts):
    """w^{r+1} = sum_j n_j / n * w_j^{r+1} (Eq. 3)."""
    weights = torch.tensor(sample_counts, dtype=torch.float32)
    weights = weights / weights.sum()
    new_state = {}
    for key in local_states[0]:
        stacked = torch.stack([s[key].float() for s in local_states])
        view = [len(local_states)] + [1] * (stacked.dim() - 1)
        new_state[key] = torch.sum(stacked * weights.to(stacked.device).view(*view), dim=0).to(local_states[0][key].dtype)
    return new_state


def aggregate_fltg(global_state, local_states, sample_counts, clip_tau=100.0, temperature=1.0, min_clients=2,
                   reference_state=None, eps=1e-12):
    """Direction-filtered aggregation of client updates.

    1) delta_j = w_j - w; 2) with a trusted reference w_0 (``reference_state``), rescale
    ||delta_j|| to ||w_0 - w||, otherwise clip it to clip_tau; 3) reference direction:
    w_0 - w, or the mean delta; 4) score_j = max(0, cos(delta_j, reference));
    5) a = softmax(score / temperature), restricted to the positive scores with a trusted reference;
    6) w_new = w + sum_j a_j delta_j. Falls back to the weighted mean with too few clients
    or when every score vanishes.
    """
    if len(local_states) < min_clients:
        return aggregate_weighted(local_states, sample_counts)

    keys = list(global_state.keys())
    float_keys = [k for k in keys if global_state[k].is_floating_point()]

    def flatten(d):
        return torch.cat([d[k].reshape(-1).float().cpu() for k in float_keys])

    reference = None
    if reference_state is not None:
        reference = {k: reference_state[k].float() - global_state[k].float() for k in keys}
        ref_norm = float(torch.norm(flatten(reference)))

    deltas = []
    for s in local_states:
        d = {k: s[k].float() - global_state[k].float() for k in keys}
        norm = float(torch.norm(flatten(d))) + eps
        scale = ref_norm / norm if reference is not None else min(1.0, clip_tau / norm)
        deltas.append({k: v * scale for k, v in d.items()} if scale != 1.0 else d)

    if reference is None:
        reference = {k: torch.stack([d[k] for d in deltas]).mean(dim=0) for k in keys}
    c_vec = flatten(reference)
    c_norm = torch.norm(c_vec) + eps
    scores = torch.stack([torch.dot(flatten(d), c_vec) / ((torch.norm(flatten(d)) + eps) * c_norm) for d in deltas])
    scores = torch.clamp(scores, min=0.0)
    if float(scores.sum()) <= 1e-8:
        return aggregate_weighted(local_states, sample_counts)

    logits = scores / max(temperature, 1e-6)
    if reference_state is not None:   # directional filtering: drop the updates opposed to the trusted reference
        logits = torch.where(scores > 0, logits, torch.full_like(scores, float('-inf')))
    a = torch.softmax(logits, dim=0)
    new_state = {}
    for k in keys:
        stacked = torch.stack([d[k] for d in deltas])
        view = [len(deltas)] + [1] * (stacked.dim() - 1)
        new_state[k] = (global_state[k].float() + torch.sum(stacked * a.to(stacked.device).view(*view), dim=0)).to(global_state[k].dtype)
    return new_state


# ─────────────────────────────── FedALA ────────────────────────────────────
class ALAAdapter:
    """Adaptive local aggregation of the higher layers (Eq. 4).

    ``begin(local_model, global_model, u_state)`` prepares a participation: the lower layers
    take the global values and the adapted parameters are snapshotted. During training,
    ``step()`` updates the weights A <- clip(A - eta * (w - u) * dL/dtheta, 0, 1).
    """

    def __init__(self, layers, eta, init):
        self.layers, self.eta, self.init = layers, eta, init
        self.weights = None

    def _is_adapted(self, name):
        return any(layer in name for layer in self.layers)

    def begin(self, local_model, global_model):
        global_params = dict(global_model.named_parameters())
        self.params, self.global_params, self.prev_params = [], [], []
        with torch.no_grad():
            for name, p in local_model.named_parameters():
                if self._is_adapted(name):
                    self.params.append(p)
                    self.global_params.append(global_params[name].detach().clone())
                    self.prev_params.append(p.detach().clone())       # u_j^r, frozen
                else:
                    p.copy_(global_params[name])                      # lower layers: global values
        if self.weights is None:
            self.weights = [torch.full_like(p, self.init) for p in self.params]

    def step(self):
        with torch.no_grad():
            for p, prev, glob, w in zip(self.params, self.prev_params, self.global_params, self.weights):
                if p.grad is None:
                    continue
                w.copy_(torch.clamp(w - self.eta * (glob - prev) * p.grad, 0.0, 1.0))

    def apply(self):
        """Initialise the adapted parameters with u + (w - u) * A (Eq. 4)."""
        with torch.no_grad():
            for p, prev, glob, w in zip(self.params, self.prev_params, self.global_params, self.weights):
                p.copy_(prev + (glob - prev) * w.clamp(0, 1))


# ─────────────────────────────── Algorithm 1 ───────────────────────────────
def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_federated(clients, model_fn, global_val, cfg, server_reference=None):
    """Algorithm 1.

    clients    : ``{name: (train_windows, val_windows)}`` from training.make_windows.
    model_fn   : callable returning a fresh backbone (see architectures.build_model).
    global_val : pooled validation windows used for the global stopping criterion.
    server_reference : optional ``(train_windows, val_windows)`` of trusted server data; with
                 FLTG, one epoch on it from the global parameters gives the reference update.
    Returns ``(global_model, history)``; the global model is frozen at stopping.
    """
    _set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = cfg.local.device
    global_model = model_fn().to(device)
    state = {name: {'model': None, 'train': None, 'fraction': None, 'ala': None} for name in clients}
    history = {'round': [], 'participants': [], 'global_score': []}
    best_score, wait = float('-inf'), 0

    for r in range(cfg.max_rounds):
        selected = [c for c in clients if cfg.participation is None or rng.random() < cfg.participation]
        if not selected:
            continue   # retain w^{r+1} = w^r

        local_states, sample_counts = [], []
        for name in selected:
            train, val = clients[name]
            st = state[name]
            first = st['model'] is None

            if first:
                # Zero-class retention fraction selected from the broadcast parameters.
                st['fraction'], local_model, st['train'], _ = select_retention_fraction(
                    deepcopy(global_model), train, val, cfg.local, cfg.retention_grid, seed=cfg.seed)
            elif cfg.algorithm == 'fedavg':
                local_model = deepcopy(global_model)
                train_local(local_model, st['train'], val, cfg.local)
            elif cfg.algorithm == 'moon':
                prev_model = deepcopy(st['model'])            # previous local model (negative reference)
                local_model = deepcopy(global_model)
                contrastive = {'prev_model': prev_model, 'global_model': deepcopy(global_model),
                               'temperature': cfg.moon_temperature, 'mu': cfg.moon_mu}
                train_local(local_model, st['train'], val, cfg.local, contrastive=contrastive)
            elif cfg.algorithm == 'fedala':
                local_model = deepcopy(st['model'])           # stored local state u_j^r
                if st['ala'] is None:
                    # First adaptation: fit the mixing weights with the full local ceiling, parameters frozen.
                    st['ala'] = ALAAdapter(cfg.ala_layers, cfg.ala_eta, cfg.ala_init)
                    st['ala'].begin(local_model, global_model)
                    u_state = deepcopy(local_model.state_dict())
                    train_local(local_model, st['train'], val, cfg.local, ala=st['ala'], update_params=False)
                    local_model.load_state_dict(u_state)
                st['ala'].begin(local_model, global_model)
                st['ala'].apply()
                # Once the weights exist, the local training call requests a single epoch.
                train_local(local_model, st['train'], val, cfg.local, max_epochs=1, ala=st['ala'])
            else:
                raise ValueError(f'Unknown algorithm {cfg.algorithm!r}')

            st['model'] = local_model
            local_states.append(deepcopy(local_model.state_dict()))
            sample_counts.append(len(train))   # n_j = |D_j|, before the zero-class subsampling

        if cfg.server_rule == 'fltg' and r > 0:
            reference_state = None
            if server_reference is not None:
                reference_model = deepcopy(global_model)
                train_local(reference_model, *server_reference, cfg.local, max_epochs=1)
                reference_state = reference_model.state_dict()
            new_state = aggregate_fltg(global_model.state_dict(), local_states, sample_counts,
                                       cfg.fltg_clip_tau, cfg.fltg_temperature, cfg.fltg_min_clients,
                                       reference_state)
        else:
            new_state = aggregate_weighted(local_states, sample_counts)
        global_model.load_state_dict(new_state)

        # Global stopping criterion on the pooled validation years.
        score = iou_score(global_val.level, predict_levels(global_model, global_val, device=device))
        history['round'].append(r)
        history['participants'].append(list(selected))
        history['global_score'].append(score)
        if score > best_score:
            best_score, wait = score, 0
        else:
            wait += 1
            if wait >= cfg.global_patience:
                break

    return global_model, history
