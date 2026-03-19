# Third-party imports
import torch
from torch import Tensor
import torch.nn as nn


class Sine(torch.nn.Module):
    """
    A custom activation function implementing the sine function.

    This module applies the sine function element-wise to the input tensor.
    It can be used as a drop-in replacement for other activation functions
    in neural network architectures.
    """

    def __init__(self) -> None:
        """
        Initialize the Sine activation function.
        """
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Apply the sine function to the input tensor.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the sine function.
        """
        return torch.sin(input)

    def __repr__(self) -> str:
        """
        Return a string representation of the Sine activation function.

        Returns:
            str: A string representation of the object.
        """
        return f"{self.__class__.__name__}()"


class ParameterizedSigmoid(nn.Module):
    def __init__(self, k=1.0, x0=0.0):
        """
        Args:
            k: Controls the steepness (smaller k = longer linear-like section)
            x0: Controls the midpoint (output = 0.5 at x=x0)
        """
        super().__init__()
        self.k = k
        self.x0 = x0

    def forward(self, x):
        return 1.0 / (1.0 + torch.exp(-self.k * (x - self.x0)))

    def approximate_linear_width(self):
        """Returns an approximation of the 'linear-like' section width"""
        # The region where the sigmoid is approximately linear is roughly ±2/k around x0
        return 4.0 / self.k
