"""
Training loop and evaluation utilities.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict


class Trainer:
    """Handles model training and evaluation with metric logging."""

    def __init__(self, model, device="cpu", lr=0.01, momentum=0.9,
                 weight_decay=0.0, optimizer_type="sgd", scheduler_type=None,
                 scheduler_tmax=None, scheduler_step_size=50,
                 scheduler_gamma=0.1):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        if optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=momentum,
                weight_decay=weight_decay)
        elif optimizer_type == "adam":
            self.optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        self.scheduler = None
        if scheduler_type == "cosine":
            t_max = scheduler_tmax or 200
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max)
        elif scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=scheduler_step_size,
                gamma=scheduler_gamma)

        self.history = defaultdict(list)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return total_loss / total, 100.0 * correct / total

    def train(self, train_loader, test_loader, epochs,
              log_interval=1, verbose=True):
        """Full training loop with logging."""
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            test_loss, test_acc = self.evaluate(test_loader)
            elapsed = time.time() - t0

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if verbose and epoch % log_interval == 0:
                print(f"Epoch {epoch:4d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                      f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
                      f"{elapsed:.1f}s")

        return dict(self.history)
