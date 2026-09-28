"""Monotonic evaluation of a frozen ordinal risk signal (Section 5, Algorithm 2).

For predicted levels S in {0,...,4} and observed counts Y, an unconstrained B-spline
response Y = f(S) + alpha_zone + gamma_date is fitted (Eq. 7) and averaged into the
adjusted mean response mu(s) (Eq. 8). Each transition (a, b) is normalised by the
positive contrast of the training-derived K-means ordinal system (Eq. 9) and the
normalised contrasts of every order k are summarised into SCORE_k (Eqs. 10-15).
Event recall (Eq. 17), false-positive rate (Eq. 18) and the support diagnostics
N_k^pairs and V_k (Eq. 16) are reported alongside.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

LEVELS = [0, 1, 2, 3, 4]

# Transition orders P_k (Eq. 10): pairs of levels separated by k.
PASSAGES = {
    1: [(0, 1), (1, 2), (2, 3), (3, 4)],
    2: [(0, 2), (1, 3), (2, 4)],
    3: [(0, 3), (1, 4)],
    4: [(0, 4)],
}


def _is_informative(values):
    """A fixed effect is kept unless it is constant or has one unique value per record."""
    n_unique = pd.Series(values).nunique()
    return 1 < n_unique < len(values)


def fit_spline_mu(levels, y, zones, dates, df_spline=5):
    """Adjusted mean response mu(s) for s = 0..4 (Eqs. 7-8).

    Returns ``(mu, fit)`` where ``mu`` maps each level to its adjusted mean response.
    """
    d = pd.DataFrame({'score': np.clip(np.asarray(levels, dtype=float), 0, 4), 'Y': np.asarray(y, dtype=float),
                      'zone': np.asarray(zones), 'date': np.asarray(dates)}).dropna()

    formula = f'Y ~ bs(score, df={df_spline}, degree=3, include_intercept=False, lower_bound=0, upper_bound=4)'
    for effect in ('zone', 'date'):
        if _is_informative(d[effect]):
            d[effect] = d[effect].astype('category')
            formula += f' + C({effect})'

    try:
        fit = smf.ols(formula, data=d).fit(cov_type='HC1')
    except Exception as e:  # too few records or a singular design
        print(f'Warning: spline fit failed: {e}')
        return {s: np.nan for s in LEVELS}, None

    # Averaging the prediction over the same support makes the fixed effects cancel in mu(b) - mu(a).
    y_max = d['Y'].max()
    if not np.isfinite(y_max) or y_max == 0:
        y_max = 1.0
    template = d.drop(columns=['score', 'Y'])
    mu = {}
    for s in LEVELS:
        pred = fit.predict(template.assign(score=float(s)))
        # Guard against a diverging extrapolation on a level with no support.
        mu[s] = 0.0 if np.any(pred > y_max * 100) else float(np.clip(pred, 0, y_max * 2.0).mean())
    return mu, fit


class MonotonicScorer:
    """Monotonic scores k1-k4 relative to a training-derived reference ordinal system.

    Usage: ``fit_reference`` on the training records with the K-means training labels as
    the reference levels, then ``score`` on the frozen predictions of the evaluated set.
    One scorer is fitted per evaluation cluster.
    """

    def __init__(self, df_spline=5, min_n=1):
        self.df_spline = df_spline
        self.min_n = min_n          # n_min: each endpoint level must be predicted at least once
        self.sigma = 1.0
        self.reference_deltas = {}  # d_ab^ref > 0 for the reference transitions

    def _mu(self, levels, y, zones, dates):
        # Y is scaled by the training standard deviation; the scale cancels in delta_ab (Eq. 9).
        return fit_spline_mu(levels, np.asarray(y, dtype=float) / self.sigma, zones, dates, self.df_spline)[0]

    def fit_reference(self, ref_levels, y, zones, dates):
        """Store the positive reference contrasts d_ab^ref of the training K-means levels."""
        y = np.asarray(y, dtype=float)
        self.sigma = float(np.std(y)) or 1.0
        mu = self._mu(ref_levels, y, zones, dates)
        counts = pd.Series(np.clip(np.asarray(ref_levels), 0, 4).astype(int)).value_counts().to_dict()
        self.reference_deltas = {}
        for pairs in PASSAGES.values():
            for a, b in pairs:
                if counts.get(a, 0) >= self.min_n and counts.get(b, 0) >= self.min_n:
                    delta = mu[b] - mu[a]
                    # Anti-monotone reference transitions are not a meaningful normalisation unit.
                    if np.isfinite(delta) and delta > 1e-9:
                        self.reference_deltas[(a, b)] = delta
        return self

    def score(self, pred_levels, y, zones, dates):
        """Return k1-k4, their support (N_k^pairs, V_k, supported flag), recall and FPR."""
        pred_levels = np.clip(np.round(np.asarray(pred_levels, dtype=float)), 0, 4).astype(int)
        y = np.asarray(y, dtype=float)
        counts = pd.Series(pred_levels).value_counts().to_dict()
        mu = self._mu(pred_levels, y, zones, dates)

        out = {}
        for k, pairs in PASSAGES.items():
            retained = [(a, b) for a, b in pairs if counts.get(a, 0) >= self.min_n and counts.get(b, 0) >= self.min_n]
            out[f'npairs_k{k}'] = len(retained)
            out[f'support_k{k}'] = int(sum(min(counts[a], counts[b]) for a, b in retained))
            out[f'supported_k{k}'] = len(retained) > 0
            if not retained or all(np.isnan(v) for v in mu.values()):
                out[f'k{k}'] = 0.0   # unsupported order: zero by reporting convention
                continue

            # Normalised contrasts (Eq. 9); a pair without positive reference contrast keeps unit scale.
            deltas = np.array([(mu[b] - mu[a]) / (abs(self.reference_deltas.get((a, b), 1.0)) or 1.0)
                               for a, b in retained])
            med, mn = np.median(deltas), np.min(deltas)
            viol = np.mean(deltas < 0.0)
            neg = np.mean(np.clip(-deltas, 0.0, None))
            score = (med + mn) / 2 - neg * (1 + viol)   # Eq. 15
            out[f'k{k}'] = 0.0 if np.isnan(score) else float(np.clip(score, -1e6, 1e6))

        positives = y > 0
        out['recall'] = float(np.mean(pred_levels[positives] > 0)) if positives.any() else 0.0   # Eq. 17
        out['fpr'] = float(np.mean(pred_levels[~positives] > 0)) if (~positives).any() else 0.0   # Eq. 18
        out['n_positive'] = int(positives.sum())
        return out


def aggregated_monotonic_score(scores):
    """Aggregate k1 + k2 + k3 + k4 used to select the zero-class retention fraction."""
    return float(sum(scores[f'k{k}'] for k in PASSAGES))


def fit_cluster_scorers(df_train, cluster_col, ref_level_col='level', target_col='nbsinister',
                        zone_col='departement', date_col='date', **kwargs):
    """One MonotonicScorer per evaluation cluster, fitted on the training years."""
    scorers = {}
    for cluster, sub in df_train.groupby(cluster_col):
        scorers[cluster] = MonotonicScorer(**kwargs).fit_reference(
            sub[ref_level_col], sub[target_col], sub[zone_col], sub[date_col])
    return scorers


def evaluate_by_cluster(df, pred_col, scorers, cluster_col, target_col='nbsinister',
                        zone_col='departement', date_col='date'):
    """Algorithm 2 on every non-empty evaluation cluster; returns one row per cluster."""
    rows = []
    for cluster, sub in df.groupby(cluster_col):
        if cluster not in scorers or sub.empty:
            continue
        res = scorers[cluster].score(sub[pred_col], sub[target_col], sub[zone_col], sub[date_col])
        rows.append({'cluster': cluster, **res})
    return pd.DataFrame(rows)
