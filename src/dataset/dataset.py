from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from src.vectorization import CountVectorizer


@dataclass
class Dataset:
    """
    Dataset for sender-receiver observations.
    Attributes
    ----------
    transition_matrix : np.ndarray
        Ground-truth sender–receiver behavior matrix with shape [n_senders x n_receivers].
    senders : np.ndarray
        Count vectors of observed senders per round, shape [num_rounds x n_senders].
        Each row represents how many times each sender was active in that round.
    receivers : np.ndarray
        Count vectors of receivers per round, shape [num_rounds x n_receivers].
        Each row represents how many times each receiver was addressed in that round.
    """
    
    transition_matrix: ndarray
    senders: ndarray
    receivers: ndarray

    def __post_init__(self):
        self._validate()
        self.transition_matrix = zero_unobserved_senders(self.transition_matrix, self.senders)

    def _validate(self):
        assert all(arr.ndim == 2 for arr in (self.transition_matrix, self.senders, self.receivers)), \
            "All inputs must be 2D arrays: got " \
            f"transition_matrix {self.transition_matrix.ndim}D, " \
            f"senders {self.senders.ndim}D, receivers {self.receivers.ndim}D."
        
        assert self.senders.shape[0] == self.receivers.shape[0], \
            f"Senders and receivers must have the same number of observations (rows), " \
            f"but got senders {self.senders.shape[0]} and receivers {self.receivers.shape[0]}."
        
        assert np.all(self.senders.sum(axis=1) == self.senders.sum(axis=1)[0]), \
            f"Batch size (total messages per round) must be consistent across all sender rows."
        
        assert np.all(self.receivers.sum(axis=1) == self.receivers.sum(axis=1)[0]), \
            f"Batch size (total messages per round) must be consistent across all receiver rows."
        
        assert self.senders.shape[1] == self.transition_matrix.shape[0], \
            f"Number of senders in data ({self.senders.shape[1]}) does not match " \
            f"transition_matrix rows ({self.transition_matrix.shape[0]})."
        
        assert self.receivers.shape[1] == self.transition_matrix.shape[1], \
            f"Number of receivers in data ({self.receivers.shape[1]}) does not match " \
            f"transition_matrix columns ({self.transition_matrix.shape[1]})."


    def __iter__(self):
        return iter((self.transition_matrix, self.senders, self.receivers))
    
    def head(self, n_obs: int):
        """
        Return a new Dataset containing only the first `n_obs` observations.

        Parameters
        ----------
        n_obs : int
            Number of observations to keep. Must be between 1 and total number of observations.

        Returns
        -------
        Dataset
            New Dataset instance with the first `n_obs` observations.
        """
        n_total = self.senders.shape[0]
        if n_obs <= 0 or n_obs > n_total:
            raise ValueError(f"n_obs must be between 1 and {n_total}, got {n_obs}.")

        return Dataset(
            transition_matrix=self.transition_matrix,
            senders=self.senders[:n_obs],
            receivers=self.receivers[:n_obs],
        )
    
    def sample(self, n_obs: int, replace: bool = False, random_state: int | None = None):
        """
        Randomly sample a subset of observations from the dataset.

        Parameters
        ----------
        n_obs : int
            Number of observations to sample.
        replace : bool, default=False
            Whether to sample with replacement.
        random_state : int or None, default=None
            Random seed for reproducibility.

        Returns
        -------
        Dataset
            New Dataset instance containing only the sampled observations.
        """
        if n_obs <= 0 or n_obs > self.senders.shape[0]:
            raise ValueError(f"n_obs must be between 1 and {self.senders.shape[0]}.")

        rng = np.random.default_rng(random_state)
        idx = rng.choice(self.senders.shape[0], size=n_obs, replace=replace)

        return Dataset(
            transition_matrix=self.transition_matrix,
            senders=self.senders[idx],
            receivers=self.receivers[idx],
        )
    
    @classmethod
    def from_batched(cls, transition_matrix: ndarray, senders: ndarray,
                     receivers: ndarray) -> "Dataset":
        vec = CountVectorizer()
        n_sender, n_receiver = transition_matrix.shape
        X = vec.transform(senders, minlen=n_sender)
        Y = vec.transform(receivers, minlen=n_receiver)

        return Dataset(transition_matrix, X, Y)
    

def zero_unobserved_senders(transition_matrix: ndarray, senders: ndarray) -> ndarray:
   """
    Zero out rows in the transition matrix corresponding to unobserved senders.

    Any sender that does not appear in the `senders` count vectors (i.e., has a
    total count of zero across all rounds) will have its entire row in the
    transition matrix set to 0. This ensures that unobserved senders do not
    contribute to the evaluation.

    Parameters
    ----------
    transition_matrix : np.ndarray
        Full sender–receiver behavior matrix of shape [n_senders x n_receivers].
    senders : np.ndarray
        Sender count matrix of shape [n_rounds x n_senders].
        Each row represents how many times each sender was active in a round.

    Returns
    -------
    np.ndarray
        Transition matrix with rows corresponding to unobserved senders set to 0.
    """
   active_senders = senders.sum(axis=0) > 0
   transition_matrix[~active_senders, :] = 0

   return transition_matrix