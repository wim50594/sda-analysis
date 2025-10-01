from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.attack.attack import AttackBase

from .torch_modules.transformer import Transformer


class TorchAttack(AttackBase):
    def __init__(self, model: nn.Module, train_params: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.train_params = train_params

    def execute(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        reset_model_weights(self.model)
        train_torch(self.model, X, Y, **self.train_params)
        device = self.train_params.get('device', 'cpu')
        # --- Aggregiertes P_avg ---
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(device)
            _, P_t_full = self.model(X_tensor)
            P_avg = P_t_full.mean(dim=0).squeeze(0).cpu().numpy()  # [N_senders, N_receivers]

        return P_avg
    
    def __str__(self) -> str:
        params_str = ', '.join(f"{k}={v}" for k, v in self.train_params.items())
        return f"{str(self.model.__class__.__name__)}({params_str})"


def train_torch(model: nn.Module, X: np.ndarray, Y: np.ndarray,
                batch_size: int = 32, epochs: int = 50, lr: float = 1e-4,
                early_stop: int | None = None, tol: float = 1e-4,
                device: str = 'cpu', verbose: bool = False, use_tqdm: bool = True):
    """
    Train a PyTorch model with optional early stopping.

    Parameters
    ----------
    model : nn.Module
        PyTorch model that returns (Y_pred, P_t) in forward pass
    X : np.ndarray, shape [M, N_senders]
    Y : np.ndarray, shape [M, N_receivers]
    batch_size : int
    epochs : int
    lr : float
    device : str
    verbose : bool
    use_tqdm : bool
    early_stop : int or None
        Stop training if validation loss does not improve for `early_stop` epochs.
        If None, no early stopping is applied.
    tol : float
        Minimum improvement to reset early stopping counter.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.to(device)

    M = X.shape[0]
    n_batches = int(np.ceil(M / batch_size))
    best_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(epochs), total=epochs, unit='epoch', disable=not use_tqdm):
        perm = np.random.permutation(M)
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i*batch_size:(i+1)*batch_size]
            X_batch = torch.from_numpy(X[idx]).float().to(device)
            Y_batch = torch.from_numpy(Y[idx]).float().to(device)

            optimizer.zero_grad()
            Y_pred, P_t = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        epoch_loss /= M

        # Early stopping check
        if early_stop is not None:
            if best_loss - epoch_loss > tol:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop and verbose:
                    print(f"Early stopping at epoch {epoch+1} (loss={epoch_loss:.6f})")
                    break

        if verbose and (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.6f}")


def reset_model_weights(model: nn.Module):
    """
    Resets all learnable parameters of a PyTorch model using its default
    initialization function (typically Xavier, Kaiming, or the module's 
    default reset method).
    """
    def weight_reset(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    model.apply(weight_reset)
