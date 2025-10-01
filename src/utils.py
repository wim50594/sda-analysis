from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from tqdm import tqdm


def get_contacts(send_behavior: ArrayLike):
    send_behavior = np.asarray(send_behavior)
    return np.where(send_behavior > 0)[0]


def target_rounds(sender: ArrayLike, target: int):
    return np.where(sender == target)[0]

def is_jagged_list(lists: Sequence) -> bool:
    lengths = [len(l) for l in lists]
    return len(set(lengths)) > 1

def sequence_to_numpy(data: Sequence[ArrayLike], pad_value: int = 0) -> np.ndarray:
    """
    Convert a sequence of array-likes (possibly of different lengths) to a 2D NumPy array.
    Shorter arrays are padded with `pad_value`.

    Parameters
    ----------
    lists : Sequence[ArrayLike]
        List of sequences (lists, arrays) to convert.
    pad_value : int or float, default 0
        Value to use for padding shorter arrays.

    Returns
    -------
    np.ndarray
        2D array with shape (len(lists), max_length), padded with `pad_value`.
    """
    data = [np.asarray(l) for l in data]
    if is_jagged_list(data):
        return np.asarray(data)

    # determine maximum length
    max_len = max(len(d) for d in data)
    
    # init padded array
    arr = np.full((len(data), max_len), pad_value, dtype=float)
    
    # fill values
    for i, l in enumerate(data):
        arr[i, :len(l)] = l

    return arr

def batched(iterable: Iterable, batch_size: int, strict: bool = False):
    """
    Yield successive batches of size `batch_size` from an iterable.

    Parameters
    ----------
    iterable : Iterable
        Any iterable (list, tuple, generator, etc.).
    batch_size : int
        Number of elements per batch.
    strict: bool
        If True, the last incomplete batch will be discarded.

    Yields
    ------
    List
        Batch of elements of length `batch_size` (last batch may be smaller or discarded).
    """
    batch: list = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch and not strict:  # yield remaining items in last batch
        yield batch