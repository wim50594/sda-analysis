import numpy as np
from numpy.typing import ArrayLike


class CountVectorizer:
    def transform(self, data: ArrayLike, minlen: int = 0):
        data = np.asarray(data)

        if data.ndim == 1:
            return np.bincount(data, minlen)
        elif data.ndim == 2:
            if not minlen:
                minlen = data.max() + 1
            return np.array([np.bincount(x, minlength=minlen) for x in data])
        else:
            raise ValueError(f"Unsupported number of dimensions: {data.ndim}. Only 1D or 2D arrays are allowed.")
    
    def invert_transform(self, counts: np.ndarray):
        """
        Invert a count-matrix (like from np.bincount) back to lists of indices.

        Parameters
        ----------
        counts : np.ndarray
            1D or 2D array of counts.

        Returns
        -------
        list of lists or list:
            Reconstructed index sequences.
        """
        counts = np.asarray(counts)
        if counts.ndim == 1:
            return np.repeat(np.arange(len(counts)), counts).tolist()
        elif counts.ndim == 2:
            return [np.repeat(np.arange(counts.shape[1]), row).tolist() for row in counts]
        else:
            raise ValueError