import numpy as np
from numpy import ndarray

from .disclosure import DisclosureAttack


class HittingSet(DisclosureAttack):
    """
    Hitting Set attack to estimate the sender–receiver transition matrix
    from batched count matrices of messages.
    """

    def _execute_targeted(self,
                          senders: ndarray,
                          receivers: ndarray,
                          senders_target: ndarray,
                          receivers_target: ndarray,
                          target_user: int) -> ndarray:
        """
        Perform a minimal hitting set attack for a specific user.

        Parameters
        ----------
        senders : ndarray
            Count matrix of senders per round, shape [n_rounds x n_senders].
        receivers : ndarray
            Count matrix of receivers per round, shape [n_rounds x n_receivers].
        senders_target : ndarray
            Sender counts for rounds where target_user is sending messages.
        receivers_target : ndarray
            Receiver counts for rounds where target_user is sending messages.
        target_user : int
            The index of the target user.

        Returns
        -------
        ndarray
            Estimated transition probabilities for the target user
            of length n_receivers.
        """
        n_receivers = receivers.shape[1]

        # Convert count matrices to sets of active receivers per round
        observations_target = [set(np.nonzero(row)[0]) for row in receivers_target]

        # Compute highest active receiver index for efficiency
        received_messages = receivers_target.sum(axis=1)
        non_zero_indices = np.nonzero(received_messages)[0]
        if len(non_zero_indices) == 0:
            return np.zeros(n_receivers)
        max_contact_index = non_zero_indices[-1] + 1

        # Search for hitting sets starting from small sizes
        contact_sets = []
        for contact_size in range(1, max_contact_index):
            contact_sets = exact_hitting_sets(observations_target, contact_size)
            if contact_sets:
                break

        # Estimate probabilities
        P_est = np.zeros(n_receivers)
        if contact_sets:
            freq = np.bincount(
                [receiver for s in contact_sets for receiver in s],
                minlength=n_receivers
            )
            P_est = freq / freq.sum()

        return P_est


def exact_hitting_sets(observations: list[set[int]],
                       max_size: int,
                       current_set: set[int] | None = None) -> list[set[int]]:
    """
    Recursive algorithm to find all hitting sets of size <= max_size.

    Parameters
    ----------
    observations : list of sets
        Each set contains receiver indices that must be intersected.
    max_size : int
        Maximum allowed size of the hitting set.
    current_set : set, optional
        Current partial hitting set being constructed.

    Returns
    -------
    list of sets
        All valid hitting sets satisfying the observations.
    """
    if current_set is None:
        current_set = set()

    # Base case: all observations are intersected
    if not observations:
        return [current_set]

    # Base case: no room to add more elements
    if max_size < 1:
        return []

    # Pick one observation to branch on
    first_obs = next(iter(observations))
    hitting_sets = []

    for receiver in first_obs:
        # Remaining observations that do not contain this receiver
        remaining_obs = [obs for obs in observations if receiver not in obs]
        for hs in exact_hitting_sets(remaining_obs, max_size - 1, current_set | {receiver}):
            hitting_sets.append(hs)

    return hitting_sets