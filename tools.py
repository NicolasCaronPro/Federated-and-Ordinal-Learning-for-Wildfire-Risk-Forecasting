import numpy as np
import pickle
from Pathlib import path
from sklearn.metrics import f1_score, recall_score, precision_score

def calculate_area_under_curve(y_values):
    """
    Calcule l'aire sous la courbe pour une série de valeurs données (méthode de trapèze).

    :param y_values: Valeurs sur l'axe des ordonnées pour calculer l'aire sous la courbe.
    :return: Aire sous la courbe.
    """
    return np.trapz(y_values, dx=1)

def iou_score(y_true, y_pred):
    """
    Calcule les scores (aire commune, union, sous-prédiction, sur-prédiction) entre deux signaux.

    Args:
        t (np.array): Tableau de temps ou indices (axe x).
        y_pred (np.array): Signal prédiction (rouge).
        y_true (np.array): Signal vérité terrain (bleu).

    Returns:
        dict: Dictionnaire contenant les scores calculés.
    """

    if isinstance(y_pred, DMatrix):
        y_pred = np.copy(y_pred.get_data().toarray())

    if isinstance(y_true, DMatrix):
        y_true = np.copy(y_true.get_label())

    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul des différentes aires
    intersection = np.trapz(np.minimum(y_pred, y_true))  # Aire commune
    union = np.trapz(np.maximum(y_pred, y_true))         # Aire d'union

    return intersection / union if union > 0 else 0

def under_prediction_score(y_true, y_pred):
    """
    Calcule le score de sous-prédiction, c'est-à-dire l'aire correspondant
    aux valeurs où la prédiction est inférieure à la vérité terrain,
    normalisée par l'union des deux signaux.

    Args:
        y_true (np.array): Signal vérité terrain.
        y_pred (np.array): Signal prédiction.

    Returns:
        float: Score de sous-prédiction.
    """

    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul de l'aire de sous-prédiction
    under_prediction_area = np.trapz(np.maximum(y_true - y_pred, 0))  # Valeurs positives où y_true > y_pred
    
    # Calcul de l'union (le maximum des deux signaux à chaque point)
    union_area = np.trapz(np.maximum(y_true, y_pred))  # Union des signaux
    
    return under_prediction_area / union_area if union_area > 0 else 0

def over_prediction_score(y_true, y_pred):
    """
    Calcule le score de sur-prédiction, c'est-à-dire l'aire correspondant
    aux valeurs où la prédiction est supérieure à la vérité terrain,
    normalisée par l'union des deux signaux.

    Args:
        y_true (np.array): Signal vérité terrain.
        y_pred (np.array): Signal prédiction.

    Returns:
        float: Score de sur-prédiction.
    """
    y_pred = np.reshape(y_pred, y_true.shape)
    # Calcul de l'aire de sur-prédiction
    over_prediction_area = np.trapz(np.maximum(y_pred - y_true, 0))  # Valeurs positives où y_pred > y_true
    
    # Calcul de l'union (le maximum des deux signaux à chaque point)
    union_area = np.trapz(np.maximum(y_true, y_pred))  # Union des signaux
    
    return over_prediction_area / union_area if union_area > 0 else 0

def evaluate_metrics(df, y_true_col='target', y_pred=None):
    """
    Calcule l'IoU et le F1-score sur chaque département, puis calcule l'aire sous la courbe normalisée (aire / aire maximale).
    
    :param dff: DataFrame contenant les colonnes ['Department', 'Scale', 'nbsinister', 'target']
    :param dataset: Nom du dataset à filtrer
    :param y_true_col: Colonne représentant les cibles réelles
    :param y_pred: Liste ou tableau des prédictions
    :param metric: Choix de la métrique ('IoU' ou 'F1')
    :param top: Nombre de départements à afficher (ou 'all' pour tout afficher)
    :return: Dictionnaire contenant l'aire normalisée pour chaque modèle.
    """
    
    # Trier les valeurs par 'nbsinister' décroissant
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

    # Initialiser un dictionnaire pour les résultats
    results = {'iou' : iou, 'f1' : f1, 'under' : under, 'over' : over, 'prec' : prec, 'recall' : rec,
               'auoc' : auoc, 'f1_macro' : f1_macro, 'prec_macro' : prec_macro, 'rec_macro' : rec_macro}

    # Calculer l'IoU et F1 pour chaque département
    IoU_scores = []
    F1_scores = []
    rec_scores = []
    prec_scores = []
    
    for i, department in enumerate(df_sorted['departement'].unique()):
        # Extraire les valeurs pour chaque département
        y_true = df_sorted[df_sorted['departement'] == department][y_true_col].values
        if np.all(y_true == 0):
            continue
        y_pred_department = y_pred[df_sorted['departement'] == department]  # Récupérer les prédictions associées au département
        
        # Calcul des scores IoU et F1
        IoU = iou_score(y_true, y_pred_department)
        F1 = f1_score(y_true > 0, y_pred_department > 0, zero_division=0)
        prec = precision_score(y_true > 0, y_pred_department > 0, zero_division=0)
        rec = recall_score(y_true > 0, y_pred_department > 0, zero_division=0)

        IoU_scores.append(IoU)
        F1_scores.append(F1)
        prec_scores.append(prec)
        rec_scores.append(rec)
        
    df_sorted_test_area = df_sorted[df_sorted[y_true_col] > 0]
    # Calcul de l'aire maximale possible (cas parfait où toutes les prédictions sont correctes)
    max_area = np.trapz(np.ones(len(df_sorted_test_area['departement'].unique())), dx=1)
    
    # Calcul de l'aire sous la courbe pour l'IoU et le F1
    IoU_area = calculate_area_under_curve(IoU_scores)
    F1_area = calculate_area_under_curve(F1_scores)

    prec_area = calculate_area_under_curve(prec_scores)
    rec_area = calculate_area_under_curve(rec_scores)

    # Normalisation par l'aire maximale
    normalized_IoU = IoU_area / max_area if max_area > 0 else 0
    normalized_F1 = F1_area / max_area if max_area > 0 else 0
    normalized_rec = rec_area / max_area if max_area > 0 else 0
    normalized_prec = prec_area / max_area if max_area > 0 else 0
    
    y_true = df[y_true_col]
    
    # Stocker les résultats dans le dictionnaire
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
    Met à jour self.metrics[tp] en stockant des tableaux NumPy.
    Pour chaque (k, v) dans metrics_run, on alimente la clé f"{k}_val".
    - v peut être un scalaire ou un array/list -> converti en 1D via np.atleast_1d.
    """
    bucket = self.metrics.setdefault(tp, {})
    for k, v in metrics_run.items():
        key = f"{k}_{set}"
        v_arr = np.atleast_1d(v).astype(float)

        if key not in bucket:
            # Première insertion -> tableau directement
            bucket[key] = v_arr.copy()
        else:
            # Concaténation avec l'existant
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
    Pour chaque clé 'metric' de d (ou sous-ensemble 'keys'), calcule l'IC95
    via calculate_ic95(d[metric]) et stocke un tuple (lower, upper) sous
    'metric{suffix}' (ex.: 'f1_ic95').

    Hypothèses:
    - d[metric] est une séquence numérique (list/tuple/ndarray) de valeurs (runs, sous-samples, etc.)
    - La fonction calculate_ic95(array_like) existe et renvoie (lower, upper)

    Paramètres
    ----------
    d : dict
        Dictionnaire des métriques => séquences de valeurs.
    keys : itérable de str, optionnel
        Si fourni, ne traite que ces clés. Sinon, toutes les clés sauf celles finissant par `suffix`.
    suffix : str
        Suffixe pour la clé IC95 (par défaut "_ic95").
    dropna : bool
        Si True, ignore les NaN avant le calcul.
    overwrite : bool
        Si False, n’écrase pas une clé '{metric}{suffix}' déjà existante.

    Retour
    ------
    dict (même objet) enrichi de paires '{metric}{suffix}': (lower, upper).
    """
    # Sélection des clés candidates
    if keys is None:
        candidates = [k for k in d.keys() if not k.endswith(suffix)]
    else:
        candidates = list(keys)

    for k in candidates:
        vals = d.get(k, None)
        if vals is None:
            continue

        # Convertir en tableau 1D de floats
        arr = np.asarray(vals, dtype=float).ravel()
        if dropna:
            arr = arr[~np.isnan(arr)]

        # Besoin d'au moins 2 points pour un IC95 basé sur SD
        if arr.size < 2:
            d[f"{k}{suffix}"] = (np.nan, np.nan)
            continue

        # Appel à la fonction externe calculate_ic95
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
    Arrondit tous les float rencontrés dans une structure Python (dict, list, tuple, set),
    et renvoie une nouvelle structure du même type.
    
    - obj: structure d'entrée (dict, list, tuple, set, scalaires)
    - ndigits: nombre de décimales (par défaut 2)
    - round_keys: si True, arrondit aussi les *clés* de type float dans les dicts
                  (attention aux collisions possibles de clés après arrondi)
    """
    # float -> on arrondit
    if isinstance(obj, float):
        return round(obj, ndigits)

    # dict -> on traite clés/valeurs
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_k = round(k, ndigits) if (round_keys and isinstance(k, float)) else k
            new_dict[new_k] = round_floats(v, ndigits, round_keys)
        return new_dict

    # list -> on traite chaque élément
    if isinstance(obj, list):
        return [round_floats(x, ndigits, round_keys) for x in obj]

    # tuple -> on traite chaque élément et on recompose un tuple
    if isinstance(obj, tuple):
        return tuple(round_floats(x, ndigits, round_keys) for x in obj)

    # set -> on traite chaque élément (attention: l'arrondi peut fusionner des éléments)
    if isinstance(obj, set):
        return {round_floats(x, ndigits, round_keys) for x in obj}

    # autre type (int, str, bool, None, etc.) -> inchangé
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
    Create a directotry if it does not exist
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

  def read_object(filename: str, path : Path):
    if not (path / filename).is_file():
        print(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))

def save_object(obj, filename: str, path : Path):
    check_and_create_path(path)
    with open(path / filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
