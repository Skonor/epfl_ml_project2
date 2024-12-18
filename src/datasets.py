from torch.utils.data import Dataset
import numpy as np
import torch
from code_generation import random_pair_generator

def generate_binary_sequence(k):
    return np.random.choice([0, 1], size=k)

class BinarySequenceDataset(Dataset):
    def __init__(self, length, sequence_length):
        """
        Args:
            length (int): Number of samples in the dataset.
            sequence_length (int): Length of each random binary sequence.
        """
        self.length = length
        self.sequence_length = sequence_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # bos = 2
        # eos = 3

        input_seq = np.concatenate([[2], generate_binary_sequence(self.sequence_length), [3]])
        output_seq = input_seq

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(output_seq, dtype=torch.long)


class RMSequenceDataset(Dataset):
    def __init__(self, length, m, r, epsilon):
        """
        Args:
            length (int): Number of samples in the dataset.
            sequence_length (int): Length of each random binary sequence.
        """
        self.length = length
        self.m = m
        self.r = r
        self.epsilon = epsilon

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # bos = 2
        # eos = 3
        # torch.manual_seed(idx)
        message, noised_message = random_pair_generator(self.m, self.r, self.epsilon)

        input_seq = np.concatenate([[2], noised_message, [3]])
        output_seq = np.concatenate([[2], message, [3]])

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(output_seq, dtype=torch.long)