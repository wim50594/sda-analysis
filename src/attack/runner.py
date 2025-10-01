import time
from collections.abc import Mapping
from typing import Any, Iterable, TypedDict

import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

from src.attack.advanced.neuronal_network.neuronal_network import attack_torch, linear_regression
from src.attack.disclosure.sda import SdaBase
from src.dataset import Dataset
from src.evaluation import evaluate
from src.typing import SupportedAttacks


class AttackConfig(TypedDict):
    """Standardized configuration dict for any attack method."""
    method: SupportedAttacks
    train_params: dict[str, Any]


AttackOrConfig = SupportedAttacks | AttackConfig


def run_attack(dataset: Dataset, method: SupportedAttacks, limit_obs: int = 0, **train_params):
    """
    Executes a single attack instance.
    """
    n_users = dataset.senders.max() + 1, dataset.receivers.max() + 1
    if limit_obs > 0:
        X, Y = dataset.X[:limit_obs], dataset.Y[:limit_obs]
    else:
        X, Y = dataset

    start = time.perf_counter()

    # 2. Dispatch based on method type
    if isinstance(method, SdaBase):
        # SDA attacks use receivers/senders directly
        P_est = method.attack(dataset.receivers, dataset.senders, limit_obs=limit_obs, **train_params)
    elif isinstance(method, nn.Module):
        # PyTorch models use the feature vectors (X, Y)
        P_est = attack_torch(method, X, Y, **train_params)
    else:
        # Sklearn-like models use the feature vectors (X, Y)
        P_est = linear_regression(method, X, Y, **train_params)

    performance = time.perf_counter() - start
    return P_est, performance


def _normalize_attack_configs(
    attacks_input: Iterable[AttackOrConfig] | dict[str, AttackOrConfig]
) -> dict[str, AttackConfig]:
    """
    Normalizes flexible attack input (Iterable or dict) into the 
    standardized dict[str, AttackConfig] format.
    """
    normalized_configs: dict[str, AttackConfig] = {}
    name_counts: dict[str, int] = {} 

    def get_unique_name(method_instance) -> str:
        """Generates a clean, unique name based on the class name."""
        base_name = type(method_instance).__name__
        
        name_counts[base_name] = name_counts.get(base_name, 0) + 1
        count = name_counts[base_name]
        return base_name if count == 1 else f"{base_name}_{count}"

    # --- Case 1: Input is already a mapping (dict[str, AttackOrConfig]) ---
    if isinstance(attacks_input, Mapping):
        # name is guaranteed to be of type str here
        for name, config_value in attacks_input.items():
            # Explicit runtime check to force the type checker to see 'name' as 'str'.
            if not isinstance(name, str):
                # Ignore entries with non-string keys, which are not expected
                continue

            if not isinstance(config_value, dict) or 'method' not in config_value:
                # Allows dictionary input like: {'SDA': Sda()}
                config_value = {'method': config_value, 'train_params': {}}

            # Here, 'config_value' is implicitly of type AttackConfig, but we handle the value
            normalized_configs[name] = {
                'method': config_value['method'],
                'train_params': config_value.get('train_params', {})
            }
        return normalized_configs

    # --- Case 2: Input is an Iterable (list/tuple of attacks or configurations) ---
    elif isinstance(attacks_input, Iterable) and not isinstance(attacks_input, str):
        for item in attacks_input:
            if isinstance(item, Mapping) and 'method' in item:
                # Item is an AttackConfig
                method = item['method']
                # Explicit type construction to fix the type error:
                config: AttackConfig = {
                    'method': method,
                    'train_params': item.get('train_params', {})
                }
            else:
                # Item is a SupportedAttacks instance
                method = item
                # Explicit type construction to fix the type error:
                config: AttackConfig = {
                    'method': method,
                    'train_params': {}
                }
            
            # Generate the name based on the method instance
            name = get_unique_name(method)
            # The error is fixed here because 'config' is now explicitly typed as AttackConfig
            normalized_configs[name] = config
        
        return normalized_configs

    # --- Error case ---
    else:
        raise TypeError("Input 'attacks' must be a dictionary or an iterable (list/tuple).")


def run_experiments(dataset: Dataset, 
                    attacks: Iterable[AttackOrConfig] | dict[str, AttackOrConfig],
                    step: int = 0) -> pd.DataFrame:
    """
    Orchestrates the evaluation by iterating through increasing observation rounds.
    """
    attacks_configs = _normalize_attack_configs(attacks)
    
    all_results: list[dict[str, Any]] = []
    P_true = dataset.P_true
    n_observations = len(dataset.receivers)

    if step <= 0:
        step = n_observations
        disable_tqdm = True
    else:
        disable_tqdm = False

    ts = np.arange(step, n_observations+1, step)

    for t in tqdm(ts, desc="Observation Rounds", unit="round", disable=disable_tqdm):
        for attack_name, attack_config in attacks_configs.items():

            # 1. Run Attack
            P_est, performance_sec = run_attack(dataset, attack_config['method'], limit_obs=t, **attack_config['train_params'])
            
            # 2. Evaluate
            metrics_mean = evaluate(P_true, P_est)
            
            # 3. Collect Results
            result = {
                'observations': t,
                'attack': attack_name,
                'runtime(s)': performance_sec,
                **metrics_mean.to_dict()
            }
            all_results.append(result)

    return pd.DataFrame(all_results)