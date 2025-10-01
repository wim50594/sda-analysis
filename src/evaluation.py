import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from rbo import RankingSimilarity
from scipy.stats.mstats import spearmanr


def pct_contacts(true_contacts: ArrayLike, est_contacts: ArrayLike):
    true_contacts = np.asarray(true_contacts)
    est_contacts = np.asarray(est_contacts)

    if len(true_contacts) == 0:
        return np.nan

    if true_contacts.ndim != est_contacts.ndim:
        raise ValueError(f"Dimensions for true_contacts ({est_contacts.ndim}) must match est_contacts ({est_contacts.ndim}).")
    
    if true_contacts.ndim == 1:
        return np.intersect1d(true_contacts, est_contacts).size / len(true_contacts)
    else:
        raise ValueError(f"Only supports 1D matrix (got {true_contacts.ndim}).")

def mean_absolute_rank_error(true_contacts: ArrayLike, est_contacts: ArrayLike) -> float:
    """
    Compute the mean absolute rank error (MARE) between two ranked lists.

    Parameters
    ----------
    true_contacts : ArrayLike, shape (n,)
        Ground truth ranking (list/array of items).
    est_contacts : ArrayLike, shape (m,)
        Estimated ranking (list/array of items).
        Must contain all items in `true_contacts`.

    Returns
    -------
    float
        Mean absolute difference in ranks.
    """
    true_contacts = np.asarray(true_contacts)
    est_contacts = np.asarray(est_contacts)

    if true_contacts.ndim != 1 or est_contacts.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    # Build mapping from item -> estimated rank
    rank_map = {val: idx for idx, val in enumerate(est_contacts)}

    # True ranks are just 0..n-1
    true_ranks = np.arange(len(true_contacts))
    est_ranks = np.array([rank_map[v] for v in true_contacts])

    # Mean absolute rank error
    return float(np.abs(true_ranks - est_ranks).mean())

def jaccard_sim(true_contacts: ArrayLike, est_contacts: ArrayLike) -> float:
    """
    Compute the Jaccard similarity between two sets of indices.

    Parameters
    ----------
    true_contacts : array-like of int
        True target indices (1D).
    est_contacts : array-like of int
        Estimated target indices (1D).

    Returns
    -------
    float
        Jaccard similarity in [0, 1].
    """
    true_contacts = np.asarray(true_contacts).ravel()
    est_contacts = np.asarray(est_contacts).ravel()

    if true_contacts.size == 0 and est_contacts.size == 0:
        return 1.0  # both empty, consider identical

    intersection = np.intersect1d(true_contacts, est_contacts)
    union = np.union1d(true_contacts, est_contacts)
    return len(intersection) / len(union)

def cosine_sim(P_true_row: ArrayLike, P_est_row: ArrayLike) -> float:
    """
    Compute cosine similarity two vectors.

    Parameters
    ----------
    P_true_row : array-like, shape (n,)
        Full vector of true sender-receiver values for one sender.
    P_est_row : array-like, shape (n,)
        Full vector of estimated sender-receiver values for the same sender.

    Returns
    -------
    float
        Cosine similarity in [0, 1].
    """
    P_true_row = np.asarray(P_true_row).ravel()
    P_est_row = np.asarray(P_est_row).ravel()

    norm_true = np.linalg.norm(P_true_row)
    norm_est = np.linalg.norm(P_est_row)

    if norm_true == 0 and norm_est == 0:
        return 1.0  # Both vectors empty, considered identical
    if norm_true == 0 or norm_est == 0:
        return 0.0

    return np.dot(P_true_row, P_est_row) / (norm_true * norm_est)



def evaluate(P_true: ArrayLike, P_est: ArrayLike):
    P_true = np.asarray(P_true)
    P_est = np.asarray(P_est)

    if P_true.ndim != P_est.ndim:
        raise ValueError(f"Dimensions for P_true ({P_true.ndim}) must match P_est ({P_est.ndim}).")
    
    if P_true.ndim == 1:
        P_true = P_true[None, :]
        P_est = P_est[None, :]
    elif P_true.ndim != 2:
        raise ValueError(f"Only supports 1D or 2D matrix (got {P_true.ndim}).")
    
    # Count of positive values per row (number of contacts)
    n_contacts = (P_true > 0).sum(axis=1)

    # Filter senders with no sender behavior.
    # This can occur because the user was never observed in participating a round.
    nonzero_rows = n_contacts > 0
    P_true = P_true[nonzero_rows]
    P_est = P_est[nonzero_rows]
    n_contacts = n_contacts[nonzero_rows]
    
    true_contacts = np.argsort(-P_true, axis=1)
    est_contacts = np.argsort(-P_est, axis=1)

    results = []

    for i in range(len(P_true)):
        result = {}
        target_contacts = true_contacts[i][:n_contacts[i]]
        est_target_contacts = est_contacts[i][:n_contacts[i]]

        result['%contacts'] = pct_contacts(target_contacts, est_target_contacts)
        result['mae'] = mean_absolute_rank_error(target_contacts, est_contacts[i])
        result['rbo'] = RankingSimilarity(target_contacts, est_contacts[i]).rbo()
        result['spearman'] = spearmanr(P_true[i], P_est[i])[0]
        # result['jaccard'] = jaccard_sim(target_contacts, est_target_contacts)
        result['cosine'] = cosine_sim(P_true[i], P_est[i])

        results.append(result)
    
    return pd.DataFrame(results).mean()