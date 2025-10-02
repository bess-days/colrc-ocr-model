from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from app.generate import entries
def make_dataset(test_size=0.2, random_state=42):
    train, val = train_test_split(entries, test_size=test_size, random_state=random_state)
    return train, val
