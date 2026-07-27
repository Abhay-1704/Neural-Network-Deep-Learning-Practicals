"""
Perceptron model used for the binary pattern classifier.

A single linear layer followed by a sigmoid activation. This is the
smallest possible neural network capable of learning a linearly
separable pattern such as an AND or OR gate.
"""

import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, in_features=2):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.linear(x)
        return self.sigmoid(z)
