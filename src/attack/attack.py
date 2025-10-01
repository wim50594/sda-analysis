from abc import ABC, abstractmethod

import numpy as np


class AttackBase(ABC):
    """
    Abstract base class for attacks on sender-receiver datasets.

    Subclasses must implement the `execute` method, which performs
    the attack and returns an estimated sender–receiver transition matrix.
    """

    @abstractmethod
    def execute(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Execute the attack on the provided dataset.

        Parameters
        ----------
        X : np.ndarray
            Count matrix of senders per round, shape [n_rounds x n_senders].
            Each row represents how many times each sender was active in that round.
        Y : np.ndarray
            Count matrix of receivers per round, shape [n_rounds x n_receivers].
            Each row represents how many times each receiver was addressed in that round.

        Returns
        -------
        np.ndarray
            Estimated sender–receiver transition matrix of shape
            [n_senders x n_receivers].
        """
        ...

    def __str__(self) -> str:
        return self.__class__.__name__