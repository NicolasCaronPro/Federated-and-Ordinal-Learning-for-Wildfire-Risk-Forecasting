import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

def calculate_area_under_curve(y_values):
    """
    Compute the area under the curve for a series of values using the trapezoidal rule.

    :param y_values: Values on the y-axis used to compute the area under the curve.
    :return: Area under the curve.
    """
    return np.trapz(y_values, dx=1)

def iou_score(y_true, y_pred):
    """
    Compute the overlap metrics (intersection, union, under-prediction, over-prediction) between two signals.

    Args:
        t (np.array): Array of timesteps or indices (x-axis).
        y_pred (np.array): Prediction signal (red).
        y_true (np.array): Ground-truth signal (blue).

    Returns:
        dict: Dictionary containing the computed scores.
    """

    if isinstance(y_pred, DMatrix):
        y_pred = np.copy(y_pred.get_data().toarray())

    if isinstance(y_true, DMatrix):
        y_true = np.copy(y_true.get_label())

    y_pred = np.reshape(y_pred, y_true.shape)
    # Compute the intersection and union areas
    intersection = np.trapz(np.minimum(y_pred, y_true))  # Shared area
    union = np.trapz(np.maximum(y_pred, y_true))         # Union area

    return intersection / union if union > 0 else 0

def under_prediction_score(y_true, y_pred):
    """
    Compute the under-prediction score, i.e., the area where the prediction
    is below the ground truth, normalized by the union of the two signals.

    Args:
        y_true (np.array): Ground-truth signal.
        y_pred (np.array): Prediction signal.

    Returns:
        float: Under-prediction score.
    """

    y_pred = np.reshape(y_pred, y_true.shape)
    # Area corresponding to under-predictions
    under_prediction_area = np.trapz(np.maximum(y_true - y_pred, 0))  # Positive values where y_true > y_pred

    # Union area (maximum of both signals at each point)
    union_area = np.trapz(np.maximum(y_true, y_pred))  # Union of the signals
    
    return under_prediction_area / union_area if union_area > 0 else 0

def over_prediction_score(y_true, y_pred):
    """
    Compute the over-prediction score, i.e., the area where the prediction
    exceeds the ground truth, normalized by the union of the two signals.

    Args:
        y_true (np.array): Ground-truth signal.
        y_pred (np.array): Prediction signal.

    Returns:
        float: Over-prediction score.
    """
    y_pred = np.reshape(y_pred, y_true.shape)
    # Area corresponding to over-predictions
    over_prediction_area = np.trapz(np.maximum(y_pred - y_true, 0))  # Positive values where y_pred > y_true

    # Union area (maximum of both signals at each point)
    union_area = np.trapz(np.maximum(y_true, y_pred))  # Union of the signals
    
    return over_prediction_area / union_area if union_area > 0 else 0

def evaluate_metrics(df, y_true_col='target', y_pred=None):
    """
    Compute IoU and F1-score for each department, then derive the normalized
    area under the curve (area / maximum area).

    :param df: DataFrame containing the columns ['Department', 'Scale', 'nbsinister', 'target']
    :param dataset: Name of the dataset to filter
    :param y_true_col: Column representing the ground-truth targets
    :param y_pred: List or array of predictions
    :param metric: Metric to compute ('IoU' or 'F1')
    :param top: Number of departments to display (or 'all' to display everything)
    :return: Dictionary containing the normalized area for each model.
    """

    # Sort values by 'nbsinister' descending if needed
    #df_sorted = df.sort_values(by='nbsinister', ascending=False)
    df_sorted = df
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 0]

    y_true = df[y_true_col]
    
    iou = iou_score(y_true, y_pred)
    f1 = f1_score((y_true > 0).astype(int), (y_pred > 0).astype(int), zero_division=0)
    prec = precision_score((y_true > 0).astype(int), (y_pred > 0).astype(int), zero_division=0)
    rec = recall_score((y_true > 0).astype(int), (y_pred > 0).astype(int), zero_division=0)
    
    f1_macro = f1_score((y_true).astype(int), (y_pred).astype(int), zero_division=0, average='macro')
    prec_macro = precision_score((y_true).astype(int), (y_pred).astype(int), zero_division=0, average='macro')
    rec_macro = recall_score((y_true).astype(int), (y_pred).astype(int), zero_division=0, average='macro')
    
    auoc = auoc_func(conf_matrix=confusion_matrix(y_true, y_pred, labels=np.union1d(y_true, y_pred)))

    under = under_prediction_score(y_true, y_pred)
    over = over_prediction_score(y_true, y_pred)

    # Initialize a dictionary to store results
    results = {'iou' : iou, 'f1' : f1, 'under' : under, 'over' : over, 'prec' : prec, 'recall' : rec,
               'auoc' : auoc, 'f1_macro' : f1_macro, 'prec_macro' : prec_macro, 'rec_macro' : rec_macro}

    # Compute IoU and F1 for each department
    IoU_scores = []
    F1_scores = []
    rec_scores = []
    prec_scores = []
    
    for i, department in enumerate(df_sorted['departement'].unique()):
        # Extract values for the current department
        y_true = df_sorted[df_sorted['departement'] == department][y_true_col].values
        if np.all(y_true == 0):
            continue
        y_pred_department = y_pred[df_sorted['departement'] == department]  # Retrieve the predictions for the department

        # Compute IoU, F1, precision, and recall
        IoU = iou_score(y_true, y_pred_department)
        F1 = f1_score(y_true > 0, y_pred_department > 0, zero_division=0)
        prec = precision_score(y_true > 0, y_pred_department > 0, zero_division=0)
        rec = recall_score(y_true > 0, y_pred_department > 0, zero_division=0)

        IoU_scores.append(IoU)
        F1_scores.append(F1)
        prec_scores.append(prec)
        rec_scores.append(rec)
        
    df_sorted_test_area = df_sorted[df_sorted[y_true_col] > 0]
    # Compute the maximum possible area (perfect predictions)
    max_area = np.trapz(np.ones(len(df_sorted_test_area['departement'].unique())), dx=1)
    
    # Area under the curve for IoU, F1, precision, and recall
    IoU_area = calculate_area_under_curve(IoU_scores)
    F1_area = calculate_area_under_curve(F1_scores)

    prec_area = calculate_area_under_curve(prec_scores)
    rec_area = calculate_area_under_curve(rec_scores)

    # Normalize by the maximum area
    normalized_IoU = IoU_area / max_area if max_area > 0 else 0
    normalized_F1 = F1_area / max_area if max_area > 0 else 0
    normalized_rec = rec_area / max_area if max_area > 0 else 0
    normalized_prec = prec_area / max_area if max_area > 0 else 0
    
    y_true = df[y_true_col]
    
    # Save normalized scores in the results dictionary
    results['normalized_iou'] = normalized_IoU
    results['normalized_f1'] = normalized_F1

    results['normalized_prec'] = normalized_prec
    results['normalized_rec'] = normalized_rec

    for elt in np.unique(y_true):
        if elt == 0:
            continue

        mask = (y_true >= elt) | (y_pred >= elt)

        if not np.any(mask):
            continue

        iou_elt = iou_score(y_true[mask], y_pred[mask])
        f1_elt = f1_score(y_true[mask] > 0, y_pred[mask] > 0, zero_division=0)
        prec_elt = precision_score(y_true[mask] > 0, y_pred[mask] > 0, zero_division=0)
        rec_elt = recall_score(y_true[mask] > 0, y_pred[mask] > 0, zero_division=0)

        f1_macro = f1_score((y_true[mask]).astype(int), (y_pred[mask]).astype(int), zero_division=0, average='macro')
        prec_macro = precision_score((y_true[mask]).astype(int), (y_pred[mask]).astype(int), zero_division=0, average='macro')
        rec_macro = recall_score((y_true[mask]).astype(int), (y_pred[mask]).astype(int), zero_division=0, average='macro')

        auoc_elt = auoc_func(confusion_matrix(y_true[mask], y_pred[mask], labels=np.union1d(y_true[mask], y_pred[mask])))

        results[f'iou_elt_sup_{elt}'] = iou_elt
        results[f'f1_elt_sup_{elt}'] = f1_elt
        results[f'prec_elt_sup_{elt}'] = prec_elt
        results[f'rec_elt_sup_{elt}'] = rec_elt
        
        results[f'f1_macro_elt_sup_{elt}'] = f1_macro
        results[f'prec_macro_elt_sup_{elt}'] = prec_macro
        results[f'rec_macro_elt_sup_{elt}'] = rec_macro

        results[f'auoc_elt_sup_{elt}'] = auoc_elt
    
    return results

def update_metrics_as_arrays(self, tp, metrics_run, set):
    """
    Update self.metrics[tp] by storing NumPy arrays.
    For each (k, v) in metrics_run, populate the key f"{k}_val".
    - v can be a scalar or an array/list -> converted to 1D via np.atleast_1d.
    """
    bucket = self.metrics.setdefault(tp, {})
    for k, v in metrics_run.items():
        key = f"{k}_{set}"
        v_arr = np.atleast_1d(v).astype(float)

        if key not in bucket:
            # First insertion -> store the array directly
            bucket[key] = v_arr.copy()
        else:
            # Concatenate with existing values
            bucket[key] = np.concatenate([bucket[key], v_arr])

from typing import Dict, Any, Iterable, Optional
import numpy as np

def add_ic95_to_dict(
    d: Dict[str, Any],
    keys: Optional[Iterable[str]] = None,
    suffix: str = "_ic95",
    dropna: bool = True,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    For each 'metric' key in d (or subset 'keys'), compute the 95% confidence
    interval via calculate_ic95(d[metric]) and store a tuple (lower, upper) under
    'metric{suffix}' (e.g., 'f1_ic95').

    Assumptions:
    - d[metric] is a numerical sequence (list/tuple/ndarray) of values (runs, sub-samples, etc.)
    - The function calculate_ic95(array_like) exists and returns (lower, upper)

    Parameters
    ----------
    d : dict
        Dictionary of metrics mapped to sequences of values.
    keys : iterable of str, optional
        If provided, only process these keys. Otherwise, process all keys except those ending with `suffix`.
    suffix : str
        Suffix for the confidence interval key (default "_ic95").
    dropna : bool
        If True, ignore NaNs before computing the interval.
    overwrite : bool
        If False, do not overwrite an existing '{metric}{suffix}' key.

    Returns
    -------
    dict (same object) enriched with '{metric}{suffix}': (lower, upper) pairs.
    """
    # Select candidate keys
    if keys is None:
        candidates = [k for k in d.keys() if not k.endswith(suffix)]
    else:
        candidates = list(keys)

    for k in candidates:
        vals = d.get(k, None)
        if vals is None:
            continue

        # Convert to a 1D float array
        arr = np.asarray(vals, dtype=float).ravel()
        if dropna:
            arr = arr[~np.isnan(arr)]

        # Need at least two points for an SD-based confidence interval
        if arr.size < 2:
            d[f"{k}{suffix}"] = (np.nan, np.nan)
            continue

        # Call the external calculate_ic95 function
        try:
            lower, upper = calculate_ic95(arr)
            lower = float(lower)
            upper = float(upper)
        except Exception:
            lower, upper = (np.nan, np.nan)

        out_key = f"{k}{suffix}"
        if overwrite or out_key not in d:
            d[out_key] = (lower, upper)

    return d

from typing import Any

def round_floats(obj: Any, ndigits: int = 2, round_keys: bool = False) -> Any:
    """
    Round every float encountered in a Python structure (dict, list, tuple, set)
    and return a new structure of the same type.

    - obj: input structure (dict, list, tuple, set, scalars)
    - ndigits: number of decimal places (default 2)
    - round_keys: if True, also round float *keys* in dictionaries
                  (beware of potential key collisions after rounding)
    """
    # Floats -> round directly
    if isinstance(obj, float):
        return round(obj, ndigits)

    # Dict -> process keys and values
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_k = round(k, ndigits) if (round_keys and isinstance(k, float)) else k
            new_dict[new_k] = round_floats(v, ndigits, round_keys)
        return new_dict

    # List -> process each element
    if isinstance(obj, list):
        return [round_floats(x, ndigits, round_keys) for x in obj]

    # Tuple -> process each element and rebuild a tuple
    if isinstance(obj, tuple):
        return tuple(round_floats(x, ndigits, round_keys) for x in obj)

    # Set -> process each element (rounding may merge elements)
    if isinstance(obj, set):
        return {round_floats(x, ndigits, round_keys) for x in obj}

    # Other types (int, str, bool, None, etc.) -> unchanged
    return obj

def auoc_func(conf_matrix: np.ndarray, n_beta: int = 1001) -> float:
    """
    Compute AUOC from a multi-class confusion matrix.

    Assumptions (faithful to the textual definition you provided):
    - Paths go from top-left (0,0) to bottom-right (K-1,K-1) with monotone moves
      (right or down), i.e., one cell is chosen per step along the grid.
    - Benefit rewards large *correct* predictions along the path:
        benefit(i,j) = p(j|i) if i == j else 0
    - Penalty increases with the (ordinal) distance between classes, weighted by
      how much mass p(j|i) sits in that cell:
        penalty(i,j) = p(j|i) * |i - j|
    - For each β in [0, 1], we define the path cost as:
        UOC_1(β) = min_path [ 1 - (sum_diag_p / K) + (β / K) * sum_penalty ]
      where K is the number of classes. AUOC is the integral of UOC_1(β) over β∈[0,1].
    - The integral is approximated numerically by the trapezoidal rule with n_beta points.

    Parameters
    ----------
    conf_matrix : np.ndarray
        Square confusion matrix of shape (K, K), counts per (true=y, pred=ŷ).
    n_beta : int
        Number of β samples (≥2). Larger -> finer integration.

    Returns
    -------
    float
        AUOC value in [0, 1+] (lower is better under this setup since cost includes 1 - benefit term).
        If you prefer a "higher is better" score, you can transform it downstream.
    """
    C = np.asarray(conf_matrix, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("conf_matrix must be a square 2D array")
    K = C.shape[0]

    # Row-normalized conditional probabilities p(ŷ | y)
    row_sums = C.sum(axis=1, keepdims=True)
    # Avoid division by zero (rows with no samples); leave zeros if a class never appears
    p = np.divide(C, np.where(row_sums == 0, 1.0, row_sums), where=row_sums != 0)

    # Benefit and penalty matrices
    benefit = np.zeros_like(p)
    np.fill_diagonal(benefit, np.diag(p))  # only reward correct cells

    # Ordinal distance |y - ŷ|
    yy, yhat = np.indices((K, K))
    penalty = p * np.abs(yy - yhat)

    # Precompute per-cell components of the additive path cost:
    # cost(i,j; β) = [1 - (benefit(i,j)/K)] + [ (β / K) * penalty(i,j) ] aggregated along the path.
    # Since "1" would be added K+K-1 times along a path if we put it per-cell, we only
    # include the *summed* terms that vary with cell and add the constant '1' once at the end:
    #   UOC_1(β) = 1 + min_path[ sum( -benefit/K + (β/K)*penalty ) ]
    neg_benefit_over_K = -benefit / K
    penalty_over_K = penalty / K

    # Dynamic programming to find min path cost for each β
    # dp[i,j](β) = min path cost to reach (i,j) = cell_cost(i,j; β) + min(dp[i-1,j], dp[i,j-1])
    # We'll compute for all β values by reusing the structure and only changing the linear term.
    betas = np.linspace(0.0, 1.0, n_beta)
    uoc_vals = np.empty_like(betas)

    for idx, beta in enumerate(betas):
        cell_cost = neg_benefit_over_K + beta * penalty_over_K

        # DP accumulate minimal sums along monotone paths
        dp = np.zeros((K, K), dtype=float)
        dp[0, 0] = cell_cost[0, 0]

        # first row
        for j in range(1, K):
            dp[0, j] = dp[0, j-1] + cell_cost[0, j]
        # first column
        for i in range(1, K):
            dp[i, 0] = dp[i-1, 0] + cell_cost[i, 0]
        # rest
        for i in range(1, K):
            for j in range(1, K):
                dp[i, j] = min(dp[i-1, j], dp[i, j-1]) + cell_cost[i, j]

        # Add the lone constant "1" term explained above
        uoc_vals[idx] = 1.0 + dp[-1, -1]

    # AUOC = ∫_0^1 UOC_1(β) dβ  (numerical integration)
    auoc_value = np.trapz(uoc_vals, betas)

    return float(auoc_value)

def check_and_create_path(path: Path):
    """
    Create a directory if it does not exist.
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def read_object(filename: str, path: Path):
    if not (path / filename).is_file():
        print(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))

def save_object(obj, filename: str, path: Path):
    check_and_create_path(path)
    with open(path / filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
