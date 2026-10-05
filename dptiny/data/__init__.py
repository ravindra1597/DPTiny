"""Data loading utilities for DPTiny."""

from dptiny.data.dataloader import DataLoader
from dptiny.data.dataset import Dataset, TensorDataset
from dptiny.data.mnist import get_mnist
from dptiny.data.fashion_mnist import get_fashion_mnist # as per assignment
from dptiny.data.transforms import Compose, Flatten, Normalize, ToFloat

__all__ = [
    "Compose",
    "DataLoader",
    "Dataset",
    "Flatten",
    "Normalize",
    "TensorDataset",
    "ToFloat",
    "get_mnist",
    "get_fashion_mnist",
]
