from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from src.dataset.dataset import Dataset


def gen_sender_behavior_fixed(n_users: int,
                              k_contacts: int | Sequence[int] | NDArray[np.int_] | None = None,
                              exact_contacts: bool = True, self_rec: bool = False) -> np.ndarray:
    """
    Generate an NxN fixed sender behavior matrix. Each row represents a sender, each column a receiver.
    Each row sums to 1 (unless k_contacts=0). Optionally, diagonal can be zero if self_rec=False.
    The sender behavior is fixed, i.e. does not change over time.

    Parameters
    ----------
    n_users : int
        Number of users (matrix size N x N)
    k_contacts : int | Sequence[int] | NDArray[np.int_] | None
        Maximum number of positive entries per row (excluding self if self_rec=False),
        or None for no limit.
    exact_contacts : bool
        If True, each row has exactly k_contacts positive values.
        If False, each row has up to k_contacts positive values
    self_rec : bool
        If True, users can send messages to themselves.
        If False, diagonal is 0.

    Returns
    -------
    np.ndarray, shape (n_users, n_users)
        Row-stochastic sender behavior matrix.
    """
    mat = np.zeros((n_users, n_users))

    if isinstance(k_contacts, int):
        if k_contacts == 0:
            return mat  # all zeros
        else:
            k_contacts = np.full(n_users, k_contacts)

    elif k_contacts is None:
        probs = np.random.random((n_users, n_users))
        if not self_rec:
            np.fill_diagonal(mat, 0)
        return probs / probs.sum(axis=1)

    if not exact_contacts:
        k_contacts = np.floor(1 + np.random.rand(len(k_contacts)) * k_contacts).astype(int)

    user_idx = np.arange(n_users)

    for sender in range(n_users):
        max_contacts = k_contacts[sender]
        probs = np.random.rand(max_contacts)
        probs /= probs.sum()

        # Determine possible receivers
        if self_rec:
            possible_idx = user_idx
        else:
            possible_idx = np.delete(user_idx, sender)

        # Pick max_contacts random positions
        if max_contacts >= len(possible_idx):
            selected_idx = possible_idx
        else:
            # np.choice is too slow
            # selected_idx = np.random.choice(possible_idx, size=max_contacts, replace=False)
            perm = np.random.permutation(possible_idx)
            selected_idx = perm[:max_contacts]

        mat[sender, selected_idx] = probs

    return mat


class GenMessagesFn(Protocol):
    def __call__(self, send_behaviors: np.ndarray, n_observations: int, batch_size: int,
                 *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        ...


def gen_messages_uniform(send_behaviors: np.ndarray, n_observations: int, batch_size: int):
    """
    Generate random messages according to sender behavior probabilities.

    Each sender is chosen among users who can send messages, and receivers
    are selected according to the sender's probability distribution.

    Parameters
    ----------
    send_behaviors : np.ndarray, shape (n_users, n_users)
        Row-stochastic matrix representing probability of each sender sending
        a message to each receiver.
    n_observations : int
        Number of observations.
    batch_size : int
        Number of messages per batch.

    Returns
    -------
    senders : np.ndarray, shape (n_observations, batch_size)
        Sender indices for each batch.
    receivers : np.ndarray, shape (n_observations, batch_size)
        Receiver indices for each batch.
    """

    n_users = send_behaviors.shape[0]
    actual_sender = np.where(send_behaviors.any(axis=1))[0]

    n_messages = n_observations * batch_size

    # Sample senders
    senders = np.random.choice(actual_sender, size=n_messages)

    # Sample receivers for each sender
    receivers = np.array([np.random.choice(range(n_users), p=send_behaviors[i]) for i in senders])

    return senders.reshape(-1, batch_size), receivers.reshape(-1, batch_size)


def gen_messages_normal(send_behaviors: np.ndarray, n_observations: int, batch_size: int, std_scale: float = 6.0):
    """
    Generate messages where senders are sampled from a normal distribution
    around each active sender index (cyclically wrapped).

    Parameters
    ----------
    send_behaviors : np.ndarray
        Row-stochastic matrix representing probability of each sender sending
        a message to each receiver.
    n_observations : int
        Number of observations.
    batch_size : int
        Number of messages per batch.
    std_scale : float
        Standard deviation scaling factor relative to number of active senders.

    Returns
    -------
    senders : np.ndarray, shape (n_observations, batch_size)
        Sender indices for each batch.
    receivers : np.ndarray, shape (n_observations, batch_size)
        Receiver indices for each batch.
    """
    n_users = send_behaviors.shape[0]
    actual_sender = np.where(send_behaviors.any(axis=1))[0]

    # Sample senders
    senders = np.asarray([np.random.normal(i, scale=len(actual_sender)/std_scale, size=batch_size).round(0) %
                         len(actual_sender) for i in range(n_observations)]).astype(int)
    senders = actual_sender[senders]

    # Sample receivers for each sender
    receivers = np.asarray([np.random.choice(range(n_users), p=send_behaviors[i]) for i in senders.reshape(-1)])
    return senders, receivers.reshape(-1, batch_size)


def get_dataset(n_user: int, batch_size: int, n_observations: int,
                k_contacts: int | Sequence[int] | NDArray[np.int_] | None = None,
                seed: int | None = None,
                gen_sender_behavior_params: dict = {},
                gen_messages: GenMessagesFn = gen_messages_uniform,
                gen_messages_params: dict = {}):
    np.random.seed(seed)
    transition_matrix = gen_sender_behavior_fixed(n_user, k_contacts=k_contacts, **gen_sender_behavior_params)
    senders, receivers = gen_messages(transition_matrix, n_observations=n_observations,
                                      batch_size=batch_size, **gen_messages_params)
    return Dataset.from_batched(transition_matrix, senders, receivers)
