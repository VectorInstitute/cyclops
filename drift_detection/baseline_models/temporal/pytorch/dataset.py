"""Dataclass for iterable dataset of inputs and targets."""
from torch.utils.data import DataLoader, TensorDataset


class Data:
    """Data class."""

    def __init__(self, inputs, target):
        """Initialize Data class."""
        self.inputs = inputs
        self.target = target

    def __getitem__(self, idx: int) -> tuple:
        """Get item for iterator.

        Parameters
        ----------
        idx: int
            Index of sample to fetch from dataset.

        Returns
        -------
        tuple
            Input and target.

        """
        return self.inputs[idx], self.target[idx]

    def __len__(self) -> int:
        """Return size of dataset, i.e. no. of samples.

        Returns
        -------
        int
            Size of dataset.

        """
        return len(self.target)

    def dim(self) -> int:
        """Get dataset dimensions (no. of features).

        Returns
        -------
        int
            Number of features.

        """
        return self.inputs.size(dim=1)

    def to_loader(self, batch_size, num_workers=0, shuffle=False, pin_memory=True):
        """Create dataloader.

        Returns
        -------
            DataLoader with input data

        """
        return DataLoader(
            TensorDataset(self.inputs, self.target),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )
