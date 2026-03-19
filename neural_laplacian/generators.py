# standard library
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

# numpy
import numpy as np

# torch
import torch


class ScalarFieldGenerator(ABC):
    def __init__(self, num_fields: Optional[int]) -> None:
        super().__init__()
        self._num_fields = num_fields

    @abstractmethod
    def generate(self, num_points: int) -> List[torch.Tensor]:
        pass


class UniformFieldGenerator(ScalarFieldGenerator):
    def __init__(self, min_val_range: Tuple[float, float], max_val_range: Tuple[float, float], **kwargs):
        super().__init__(**kwargs)
        self.min_val_range = min_val_range
        self.max_val_range = max_val_range

    def generate(self, num_points: int) -> List[torch.Tensor]:
        scalar_fields = []
        for _ in range(self._num_fields):
            if self.max_val_range is not None:
                min_val = np.random.uniform(*self.min_val_range)
                max_val = np.random.uniform(*self.max_val_range)
            else:
                min_val = self.min_val_range[0]
                max_val = self.min_val_range[1]
            scalar_field = torch.rand(num_points, 1) * (max_val - min_val) + min_val
            scalar_fields.append(scalar_field)
        return scalar_fields
