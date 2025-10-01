from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset.dataset import Dataset


def read_dataframe(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    chats = pd.read_pickle(path)
    messages = pd.concat(chats['messages'].tolist(), keys=chats.index)
    messages['chat_id'] = messages.index.get_level_values(0)

    return chats, messages


def filter_most_active_months(df: pd.DataFrame, nlargest: int):
    activity_month = df.groupby(df['date'].dt.to_period('M')).size()
    top_months = activity_month.nlargest(nlargest).index
    return df[df['date'].dt.to_period('M').isin(top_months)].sort_values('date')


def sample_messages_by_user(df: pd.DataFrame, n: int, random_state: int | None = None):
    unique_users = df['user'].unique()
    sample_users = pd.Series(unique_users).sample(n=n, random_state=random_state).tolist()
    sampled_messages = df[df['user'].isin(sample_users)]
    return sampled_messages


def merge_user(df: pd.DataFrame, group_size: int, exact_group: bool = True, random_state: int | None = None) -> pd.DataFrame:
    """
    Merges users into groups.

    Parameters:
    - df: DataFrame with a column 'user'
    - group_size: maximum group size
    - exact_group: if True, all groups have exactly group_size (last group may be smaller)
                   if False, groups have random size up to group_size
    - random_state: seed for reproducibility
    """
    df = df.rename(columns={'user': 'original_user'})
    np.random.seed(random_state)

    users = df['original_user'].unique()
    np.random.shuffle(users)

    group_ids = {}
    group_id = 0
    i = 0
    n = len(users)

    while i < n:
        if exact_group:
            # Gruppengröße fix
            current_group_size = min(group_size, n - i)
        else:
            # Zufällige Gruppengröße bis max group_size
            remaining = n - i
            current_group_size = np.random.randint(1, min(group_size, remaining) + 1)

        for user in users[i:i+current_group_size]:
            group_ids[user] = group_id

        group_id += 1
        i += current_group_size

    df['user'] = df['original_user'].map(group_ids)
    return df


def prepare_dataset(path: str | Path, most_active_months: int = 0,
                    sample_user: int = 0, group_size: int = 0,
                    exact_grouping: bool = True, random_state: int | None = None):

    chats, messages = read_dataframe(path)

    if most_active_months > 0:
        messages = filter_most_active_months(messages, most_active_months)

    if group_size > 0:
        messages = merge_user(messages, group_size, exact_group=exact_grouping, random_state=random_state)

    if sample_user > 0:
        messages = sample_messages_by_user(messages, sample_user, random_state)

    messages['user'], _ = pd.factorize(messages['user'])
    messages['chat_id'], _ = pd.factorize(messages['chat_id'])

    return chats, messages


def batched(df: pd.DataFrame, batch_size: int, col: str):
    num_batches = len(df) // batch_size
    truncate_len = num_batches * batch_size

    return df[col].to_numpy()[:truncate_len].reshape(num_batches, batch_size)


def get_dataset(path: str, batch_size: int, n_user: int = 0, n_observations: int = 0,
                k_contacts: int = 0, most_active_months: int = 0, seed: int | None = None):
    _, observations = prepare_dataset(path, most_active_months, group_size=k_contacts,
                                      sample_user=n_user, random_state=seed)
    senders = batched(observations, batch_size, 'user')
    receivers = batched(observations, batch_size, 'chat_id')
    if n_observations > 0:
        senders = senders[:n_observations]
        receivers = receivers[:n_observations]
    transition_matrix = pd.crosstab(observations['user'], observations['chat_id'], normalize='index').to_numpy()

    return Dataset.from_batched(transition_matrix, senders, receivers)
