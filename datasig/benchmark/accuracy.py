from torchvision.datasets import MNIST
from datasig.dataset import (
    CanonicalDataset,
    Dataset,
    TorchVisionDataset,
    ARFFDataset,
    DatasetType,
)
from datasig.config import *
import numpy as np
import torch
from typing import Any
from .utils import FingerprintMethod, extract_arff_indices
import tempfile


@dataclass
class AccuracyConfig:
    min_subset_size: int = 1000
    max_subset_size: int = 3000
    n_samples: int = 10


class FingerprintAccuracyRandomTester:
    """
    This class is used to test the accuracy of a fingerprinting method on a given
    dataset.
    It generates random subsets of the dataset and computes the true Jaccard
    similarity between them. It then computes the estimated similarity using
    the fingerprint method and compares the two.
    The `min_subset_size` and `max_subset_size` parameters specify how big the
    subsets can be.
    The `n_samples` parameter specifies how many random pairs of subsets to test.
    """

    def __init__(
        self,
        dataset: Dataset,
        fingerprint_method: FingerprintMethod,
        fingerprint_config: Any,
        config: AccuracyConfig,
    ) -> None:
        self.dataset = dataset
        self.fingerprint_method = fingerprint_method
        self.fingerprint_config = fingerprint_config
        self.config = config

    def _get_random_data_indices(self):
        """Extract a random subset of indices from the dataset"""
        subset_size = np.random.randint(
            min(len(self.dataset), self.config.min_subset_size),
            min(len(self.dataset), self.config.max_subset_size),
        )
        indices: np.ndarray = np.arange(len(self.dataset))
        np.random.shuffle(indices)
        return indices[:subset_size]

    def _get_random_torch_subset(self) -> tuple[set[int], torch.utils.data.Subset]:
        """Extract a random subset from a torch dataset"""
        indices = self._get_random_data_indices()
        return set(indices), torch.utils.data.Subset(self.dataset.dataset, indices)

    def _get_random_torch_subsets(self) -> tuple[CanonicalDataset, CanonicalDataset, float]:
        """Extract two random subsets from a torch dataset and compute their true
        Jaccard similarity"""
        indices_a, subset_a = self._get_random_torch_subset()
        indices_b, subset_b = self._get_random_torch_subset()

        jaccard = len(indices_a.intersection(indices_b)) / len(indices_a.union(indices_b))

        c_subset_a = CanonicalDataset(TorchVisionDataset(subset_a), config=self.fingerprint_config)
        c_subset_b = CanonicalDataset(TorchVisionDataset(subset_b), config=self.fingerprint_config)

        return c_subset_a, c_subset_b, jaccard

    def _gen_arff_subsets(
        self, indices_a: set[int], indices_b: set[int]
    ) -> tuple[list[str], list[str]]:
        return subset_a, subset_b

    def _get_random_arff_subsets(self) -> tuple[CanonicalDataset, CanonicalDataset, float]:
        """Extract two random subsets from a csv dataset and compute their true
        Jaccard similarity"""
        indices_a = set(self._get_random_data_indices())
        indices_b = set(self._get_random_data_indices())
        jaccard = len(indices_a.intersection(indices_b)) / len(indices_a.union(indices_b))

        file_a = tempfile.NamedTemporaryFile(suffix=".arff").name
        file_b = tempfile.NamedTemporaryFile(suffix=".arff").name
        extract_arff_indices(self.dataset.arff_file, indices_a, file_a)
        extract_arff_indices(self.dataset.arff_file, indices_b, file_b)
        subset_a = CanonicalDataset(ARFFDataset(file_a), config=self.fingerprint_config)
        subset_b = CanonicalDataset(ARFFDataset(file_b), config=self.fingerprint_config)

        return subset_a, subset_b, jaccard

    def run(self) -> float:
        """Measure the accuracy of the fingerprint method. This function
        generates fingerprints for random subsets of the supplied dataset
        and compares their estimated similarity to the true similarity.

        It returns the average error over the number of tests. The lesser the better."""

        results = [self.one_random_accuracy_test() for _ in range(self.config.n_samples)]
        # Return average error
        return sum(results) / len(results)

    def one_random_accuracy_test(self) -> float:
        """Run a single random accuracy test.

        This returns the distance between estimated similarity and true
        similarity. In other words it returns the error in estimated
        similarity. The bigger the return value is the less accurate the
        estimated similarity is"""
        # TODO(boyan): support remaining dataset formats
        if self.dataset.type == DatasetType.TORCH:
            subset_a, subset_b, jaccard = self._get_random_torch_subsets()
        elif self.dataset.type == DatasetType.ARFF:
            subset_a, subset_b, jaccard = self._get_random_arff_subsets()
        else:
            raise NotImplemented(f"Unsupported dataset type: {self.dataset.type}")
        fingerprint_a = self.fingerprint_method(subset_a)
        fingerprint_b = self.fingerprint_method(subset_b)
        return abs(fingerprint_a.similarity(fingerprint_b) - jaccard)
