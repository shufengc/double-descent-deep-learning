"""
Neural network models with configurable width and depth for studying double descent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron with configurable width and depth."""

    def __init__(self, input_dim, num_classes, hidden_width, num_hidden_layers=1):
        super().__init__()
        layers = []
        in_features = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_width))
            layers.append(nn.ReLU())
            in_features = hidden_width
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """ResNet with configurable width multiplier k (supports fractional k)."""

    def __init__(self, num_classes=10, k=1):
        super().__init__()
        k16 = max(1, int(16 * k))
        k32 = max(1, int(32 * k))
        k64 = max(1, int(64 * k))
        self.in_planes = k16
        self.conv1 = nn.Conv2d(3, k16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(k16)
        self.layer1 = self._make_layer(k16, 2, stride=1)
        self.layer2 = self._make_layer(k32, 2, stride=2)
        self.layer3 = self._make_layer(k64, 2, stride=2)
        self.linear = nn.Linear(k64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class CNN(nn.Module):
    """Simple CNN with configurable number of filters for width sweeps."""

    def __init__(self, num_classes=10, num_filters=16, input_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters, num_filters * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(num_filters * 2, num_filters * 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(num_filters * 4 * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
