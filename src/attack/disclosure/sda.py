import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike

from src.dataset import Dataset

from .disclosure import DisclosureAttack, targeted_rounds


class Sda(DisclosureAttack):
    """
    Vanilla Sender-Receiver Disclosure Attack (SDA).

    This attack estimates the target user's sender–receiver transition probabilities
    based on observed message counts and subtracts a constant noise term from all receivers.
    """

    def _execute_targeted(self, senders: ndarray, receivers: ndarray,
                          senders_target: ndarray, receivers_target: ndarray,
                          target_user: int) -> ndarray:
        # Count messages per round
        batch_size = receivers_target[0].sum()

        freqs = receivers_target.sum(axis=0)
        noise = np.full(freqs.size, 1/freqs.size)
        target_prop = freqs / freqs.size - (batch_size - 1)*noise
        return target_prop


class SdaCount(DisclosureAttack):
    """
    Simple count-based variant of the Sender-Receiver Disclosure Attack (SDA).

    This variant directly uses raw counts of messages received by each receiver 
    for the targeted user, without applying the normalization used in the vanilla SDA.
    The vanilla SDA subtracts a constant noise term from all receivers, 
    which is unnecessary when only relative frequencies matter.

    The attack returns a probability-like vector proportional to the total 
    number of messages each receiver received from the targeted user.
    """

    def _execute_targeted(self, senders: ndarray, receivers: ndarray,
                          senders_target: ndarray, receivers_target: ndarray,
                          target_user: int) -> ndarray:

        freqs = receivers_target.sum(axis=0)
        target_prop = freqs / freqs.sum()
        return target_prop


class SdaRandom(DisclosureAttack):
    """
    Randomized Sender-Receiver Disclosure Attack (SDA variant).

    This variant assigns random transition probabilities to receivers for a 
    targeted user, while ensuring that only receivers who actually received 
    messages in the observed rounds are considered.
    """

    def _execute_targeted(self, senders: ndarray, receivers: ndarray,
                          senders_target: ndarray, receivers_target: ndarray,
                          target_user: int) -> ndarray:
        total_received = receivers_target.sum(axis=0)
        has_received = total_received > 0

        target_prop = np.zeros(receivers_target.shape[1])
        # Assign random values only to receivers that received any messages
        target_prop[has_received] = np.random.random(np.sum(has_received))
        return target_prop / target_prop.sum()


class SdaSn(DisclosureAttack):
    """
    SDA variant that estimates the SDA "noise" using weighted co-sender behavior.
    """

    def _execute_targeted(self, senders: ndarray, receivers: ndarray,
                        senders_target: ndarray, receivers_target: ndarray,
                        target_user: int) -> ndarray:

        n_receivers = receivers_target.shape[1]
        # Determine the batch_size (total messages per round)
        batch_size = receivers_target[0].sum()

        freqs = receivers_target.sum(axis=0)

        # Total messages sent by each sender across targeted rounds
        total_senders = senders_target.sum(axis=0)
        
        weights_cosender = total_senders
        weights_cosender[target_user] = 0
        weights_cosender = weights_cosender / weights_cosender.sum()

        sender_indices = np.nonzero(total_senders > 0)[0]
        cosenders = sender_indices[sender_indices != target_user]
        behaviors_cosender = np.zeros((weights_cosender.size, n_receivers))
        behaviors_cosender[cosenders] = np.asarray([
            get_cosender_behavior(senders, receivers, cosender)
            for cosender in cosenders
        ])
        weights_cosender = np.expand_dims(weights_cosender, 0)
        noise = np.matmul(weights_cosender, behaviors_cosender).squeeze()

        target_prop = freqs / freqs.size - (batch_size - 1)*noise
        return target_prop


def get_cosender_behavior(senders: ndarray, receivers: ndarray, cosender: int):
    senders_target, receivers_target = targeted_rounds(senders, receivers, cosender)
    target_counts = senders_target[:, cosender]  # shape: (n_rounds,)

    # Weight the receiver counts by how many times the cosender sent messages in that round
    # Each row of receivers_target is multiplied by the corresponding count in target_counts
    receivers_target = receivers_target * target_counts[:, np.newaxis]
    target_prop = receivers_target.sum(axis=0)
    return target_prop / target_prop.sum()
