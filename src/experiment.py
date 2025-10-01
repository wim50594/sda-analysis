import time
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.attack import AttackBase
from src.dataset import Dataset
from src.evaluation import evaluate


def run_experiments(dataset: Dataset, attacks: Iterable[AttackBase] | AttackBase,
                    step: int = 0, repeat: int = 1) -> pd.DataFrame:
    """
    Orchestrates evaluation by iterating through increasing observation rounds,
    repeating each experiment multiple times and averaging results per attack.

    Parameters
    ----------
    dataset : Dataset
        Dataset object containing senders, receivers, and transition_matrix.
    attacks : AttackBase or iterable of AttackBase
        Attack(s) to evaluate.
    step : int, default=0
        Step size for increasing observation rounds.
        If 0, use the full dataset in one step.
    repeat : int, default=5
        Number of repetitions per attack/observation count.

    Returns
    -------
    pd.DataFrame
        Mean evaluation metrics for each attack at each observation count.
    """
    
    n_observations = len(dataset.receivers)
    repeat = max(repeat, 1)
    step = step if step > 0 else n_observations
    ts = np.arange(step, n_observations + 1, step)
    disable_tqdm = step == n_observations

    if isinstance(attacks, AttackBase):
        attacks = [attacks]

    all_results: list[pd.DataFrame] = []
    for t in tqdm(ts, desc="Observation Rounds", unit="round", disable=disable_tqdm):
        repeat_results = []
        for _ in range(repeat):
            dataset_sample = dataset.sample(t)
            P_true = dataset_sample.transition_matrix
            for attack in attacks:
                attack_name = str(attack)

                # 1. Run Attack
                start_time = time.perf_counter()
                P_est = attack.execute(dataset_sample.senders, dataset_sample.receivers)
                runtime_seconds = time.perf_counter() - start_time
            
                # 2. Evaluate
                metrics_mean = evaluate(P_true, P_est)
            
                # 3. Collect repeat results
                repeat_results.append({
                    'observations': t,
                    'attack': attack_name,
                    'runtime(s)': runtime_seconds,
                    **metrics_mean.to_dict()
                })
            
        # 4. Average over repeats per attack
        df_repeat = pd.DataFrame(repeat_results)
        df_avg = df_repeat.groupby(['observations', 'attack'], as_index=False).mean()
        all_results.append(df_avg)

    return pd.concat(all_results, ignore_index=True)