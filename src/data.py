"""
Data loading utilities with support for label noise and subset sampling.
"""


import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


def get_cifar10(data_dir="./data", augment=True):
    """Load CIFAR-10 with optional data augmentation."""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616))
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)

    return train_set, test_set


def get_mnist(data_dir="./data"):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def corrupt_labels(dataset, noise_rate, num_classes=10, seed=42):
    """
    Corrupt a fraction of labels uniformly at random.
    Returns the dataset with modified targets.
    """
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    n = len(targets)
    num_corrupt = int(noise_rate * n)
    corrupt_indices = rng.choice(n, size=num_corrupt, replace=False)
    for idx in corrupt_indices:
        old_label = targets[idx]
        new_label = rng.choice([l for l in range(num_classes) if l != old_label])
        targets[idx] = new_label
    dataset.targets = targets.tolist()
    return dataset


def make_subset(dataset, n_samples, seed=42):
    """Return a random subset of the dataset."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)
    return Subset(dataset, indices)


def make_loaders(train_set, test_set, batch_size=128, num_workers=0,
                 pin_memory=None):
    """Create DataLoader objects."""
    if pin_memory is None:
        import torch
        pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0)
    return train_loader, test_loader
