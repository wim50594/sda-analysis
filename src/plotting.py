import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Setup for consistent plotting
sns.set_theme(style="whitegrid")

def plot_evaluation_results(results_df: pd.DataFrame,
                            metrics_idx: list[int] | None = None,
                            ncols: int = 3,
                            simple_names: bool = True,
                            logy: bool | list[int] = False):
    """
    Plots the different evaluation metrics over increasing observation rounds.
    
    Args:
        results_df: A DataFrame with columns 'rounds', 'attack', and metric columns.
        output_path: Path to save the plot.
    """
    if results_df.empty:
        raise ValueError("Results DataFrame is empty. Cannot plot.")

    # Simplify attack names
    if simple_names and 'attack' in results_df.columns:
        results_df['attack'] = results_df['attack'].astype(str).str.replace(r'\(.*\)', '', regex=True)

    # Get a list of all metric columns
    metric_cols = [col for col in results_df.columns if col not in ['observations', 'attack']]

    if metrics_idx is not None:
        # Keep only selected metrics
        metric_cols = [metric_cols[i] for i in metrics_idx if i < len(metric_cols)]
    
    n_metrics = len(metric_cols)
    if n_metrics == 0:
        raise ValueError("Warning: No metrics found in DataFrame.")

    # Determine grid size (e.g., 2 columns)
    nrows = (n_metrics + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5))
    axes = axes.flatten()

    for i, metric in enumerate(metric_cols):
        sns.lineplot(
            data=results_df, 
            x='observations', 
            y=metric, 
            hue='attack', 
            marker='o', 
            ax=axes[i],
        )
        axes[i].set_xlabel('Observation Rounds')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].legend(title='Attack Method', loc='best')

    # Hide any unused subplots
    for i in range(len(metric_cols) + 1, len(axes)):
        fig.delaxes(axes[i])

    if logy:
        if isinstance(logy, bool):
            logy = list(range(n_metrics))
        for i in logy:
            axes[i].set_yscale('log')

    plt.tight_layout()
    
    return fig, axes

def plot_sender_receiver_distribution(sender: np.ndarray, receiver: np.ndarray):
    count_sender = sender.sum(axis=0) / sender.sum()
    count_receiver = sender.sum(axis=0) / receiver.sum()

    fig, axs = plt.subplots(2, 1, figsize=(14, 6))
    pd.Series(count_sender).plot.bar(ax=axs[0], title="Sender")
    pd.Series(count_receiver).plot.bar(ax=axs[1], title="Receiver")

    plt.tight_layout()

    return fig, axs


def plot_user_participation(senders: np.ndarray, receivers: np.ndarray):
    """
    Plots scatter plots for senders and receivers per round.

    Args:
        senders: np.ndarray of shape (n_rounds, n_users)
        receivers: np.ndarray of shape (n_rounds, n_users)

    Returns:
        fig, axs: matplotlib figure and axes
    """
    def get_xy(matrix):
        """Helper to convert matrix to x, y coordinates (one dot per user per round)."""
        n_rounds = matrix.shape[0]
        x, y = [], []
        for r in range(n_rounds):
            active_users = np.where(matrix[r] > 0)[0]
            x.extend([r] * len(active_users))
            y.extend(active_users)
        return x, y

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    fig.suptitle("User participation")

    # Plot senders
    x, y = get_xy(senders)
    sns.scatterplot(x=x, y=y, s=14, edgecolor="black", alpha=0.7, ax=axs[0])
    axs[0].set_xlabel("Round")
    axs[0].set_ylabel("User")
    axs[0].set_title("Senders")
    axs[0].locator_params(axis='y', nbins=10)
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Plot receivers
    x, y = get_xy(receivers)
    sns.scatterplot(x=x, y=y, s=14, edgecolor="black", alpha=0.7, ax=axs[1])
    axs[1].set_xlabel("Round")
    axs[1].set_title("Receivers")
    axs[1].locator_params(axis='y', nbins=10)
    axs[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig, axs