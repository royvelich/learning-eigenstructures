# standard library
from typing import List
from dataclasses import dataclass

# lightning
import lightning
import pytorch_lightning as pl

# torch geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset


@dataclass
class DatasetSpecification:
    """Configuration class for dataset loading parameters.

    Attributes:
        dataset: The PyTorch Geometric dataset
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
        shuffle: Whether to shuffle the data
    """
    dataset: Dataset
    batch_size: int
    num_workers: int
    shuffle: bool
    persistent_workers: bool
    pin_memory: bool
    drop_last: bool

class DataModule(lightning.pytorch.LightningDataModule):
    """PyTorch Lightning DataModule for handling PyTorch Geometric datasets.

    This module manages the creation of train and validation dataloaders
    for PyTorch Geometric data, supporting multiple validation datasets.

    Attributes:
        _train_dataset_specification: Specification for the training dataset
        _val_dataset_specifications: List of specifications for validation datasets
    """

    def __init__(
            self,
            train_dataset_specification: DatasetSpecification,
            val_dataset_specifications: List[DatasetSpecification],
    ) -> None:
        """Initialize the DataModule.

        Args:
            train_dataset_specification: Configuration for training dataset
            val_dataset_specifications: List of configurations for validation datasets
        """
        super().__init__()
        self._train_dataset_specification = train_dataset_specification
        self._val_dataset_specifications = val_dataset_specifications

    def train_dataloader(self) -> DataLoader:
        """Create and return the training dataloader.

        Returns:
            DataLoader configured according to the training specification
        """
        return DataLoader(
            dataset=self._train_dataset_specification.dataset,
            batch_size=self._train_dataset_specification.batch_size,
            shuffle=self._train_dataset_specification.shuffle,
            num_workers=self._train_dataset_specification.num_workers,
            persistent_workers=self._train_dataset_specification.persistent_workers,
            # prefetch_factor=5 * self._train_dataset_specification.batch_size,
            # multiprocessing_context='spawn',
            pin_memory=self._train_dataset_specification.pin_memory,
            drop_last=self._train_dataset_specification.drop_last
        )

    def val_dataloader(self) -> List[DataLoader]:
        """Create and return the validation dataloaders.

        Returns:
            List of DataLoaders, one for each validation dataset specification
        """
        return [
            DataLoader(
                dataset=spec.dataset,
                batch_size=spec.batch_size,
                shuffle=spec.shuffle,
                num_workers=spec.num_workers,
                persistent_workers=spec.persistent_workers,
                # multiprocessing_context='spawn',
                pin_memory=spec.pin_memory,
                drop_last=spec.drop_last
                # prefetch_factor=5*spec.batch_size
            ) for spec in self._val_dataset_specifications
        ]