from abc import abstractmethod

import numpy as np
from numpy import ndarray

from src.attack.attack import AttackBase


class DisclosureAttack(AttackBase):
    def execute(self, X: ndarray, Y: ndarray) -> ndarray:
        n_senders = X.shape[1]
        
        return np.array([self.execute_targeted(X, Y, t) for t in range(n_senders)])
    
    def execute_targeted(self, senders: ndarray, receivers: ndarray, target_user: int,
                         limit_obs: int = 0) -> ndarray:
        """
        Executes a targeted disclosure attack for a specific user. It filters the rounds to only those sent by `target_user`.
        Parameters
        ----------
        senders : np.ndarray
            Count matrix of senders per round, shape [n_rounds x n_senders].
            Each row represents how many times each sender was active in that round.
        receivers : np.ndarray
            Count matrix of receivers per round, shape [n_rounds x n_receivers].
            Each row represents how many times each receiver was addressed in that round.
        target_user : int, optional
            Index of the target user. If not provided, assumes the user participated in all rounds.
        limit_obs : int, optional
            Limit the number of observations (rounds) to consider. If 0, uses all observations.
        Returns
        -------
        np.ndarray
            Estimated sender behavior for the target user.
        """

        senders_target, receivers_target = targeted_rounds(senders, receivers, target_user)

        if limit_obs > 0:
            receivers_target = receivers_target[:limit_obs]
            senders_target = senders_target[:limit_obs]

        if senders_target.size == 0:
            return np.zeros(receivers.shape[1])
        
        return self._execute_targeted(senders, receivers, senders_target, receivers_target, target_user)

    @abstractmethod
    def _execute_targeted(self, senders: ndarray, receivers: ndarray,
                          senders_target: ndarray, receivers_target: ndarray,
                          target_user: int) -> ndarray:
        ...

def targeted_rounds(senders: ndarray, receivers: ndarray, target_user: int):
    """
    Filter rounds where `target_user` sent.
    Returns senders_target and receivers_target.
    """

    rounds_target = senders[:, target_user] > 0
    senders_target = senders[rounds_target]
    receivers_target = receivers[rounds_target]

    return senders_target, receivers_target