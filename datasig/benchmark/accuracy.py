from datasig.dataset import (
    TorchVisionDataset,
    ARFFDataset,
)
from datasig.algo import Algorithm
from datasig.fingerprint import DatasetFingerprint
import numpy as np
import torch
from typing import Optional
from .utils import extract_arff_indices
import tempfile
from .logger import logger
from PIL.Image import Image
from collections.abc import Sized
from dataclasses import dataclass


@dataclass
class AccuracyConfig:
    min_subset_size: int = 1000
    max_subset_size: int = 3000
    n_samples: int = 10


class SubsetCache:
    """A class used to manage subsets of various datasets being tested"""

    def __init__(self, accuracy_config: AccuracyConfig):
        self.config = accuracy_config
        self.subsets = dict()
        self.subset_indices = dict()

    def clean(self):
        """Clean the cache"""
        self.subsets: dict[
            str,
            list[
                tuple[TorchVisionDataset, TorchVisionDataset, float]
                | tuple[ARFFDataset, ARFFDataset, float]
            ],
        ] = dict()
        self.subset_indices: dict[str, int] = dict()

    def reset(self):
        """Reset the indices to iterate over the cached subsets.
        Note that this doesn't clean the cache"""
        for dataset_name in self.subset_indices.keys():
            self.subset_indices[dataset_name] = 0

    def get_subset_pair(
        self, dataset: TorchVisionDataset | ARFFDataset
    ) -> (
        tuple[TorchVisionDataset, TorchVisionDataset, float]
        | tuple[ARFFDataset, ARFFDataset, float]
    ):
        """Return a tuple (subset_a, subset_b, true_jaccard).
        The subsets are CanonicalDataset objects"""
        idx = self.subset_indices.get(dataset.name, None)
        if idx is None:
            # Seeing this dataset for the first time, init
            # data structures accordingly
            idx = 0
            self.subsets[dataset.name] = []

        if idx == len(self.subsets[dataset.name]):
            # Need to generate new subsets pair
            # TODO(boyan): support remaining dataset formats
            logger.debug("Generating new subset pair for dataset %s", dataset.name)
            if isinstance(dataset, TorchVisionDataset):
                subsets = self._get_random_torch_subsets(dataset)
            else:
                subsets = self._get_random_arff_subsets(dataset)
            self.subsets[dataset.name].append(subsets)
        else:
            logger.debug(
                "Using cached subset pair for dataset %s, cache index %d", dataset.name, idx
            )
        # Increment index of the subset pair to get next time
        self.subset_indices[dataset.name] = idx + 1
        # Return the subset pair
        return self.subsets[dataset.name][idx]

    def _get_random_data_indices(self, dataset: Sized) -> set[int]:
        """Extract a random subset of indices from the dataset"""
        subset_size = np.random.randint(
            min(len(dataset), self.config.min_subset_size),
            min(len(dataset), self.config.max_subset_size),
        )
        indices: np.ndarray = np.arange(len(dataset))
        np.random.shuffle(indices)
        return set(indices[:subset_size])

    def _get_random_torch_subset(
        self, dataset: TorchVisionDataset
    ) -> tuple[set[int], torch.utils.data.Subset[tuple[Image, int]]]:
        """Extract a random subset from a torch dataset"""
        indices = self._get_random_data_indices(dataset)
        if isinstance(dataset.dataset, torch.utils.data.Dataset):
            subset: torch.utils.data.Subset[tuple[Image, int]] = torch.utils.data.Subset(
                dataset.dataset, list(indices)
            )  # pyright: ignore[reportUnknownVariableType]
        else:
            raise NotImplementedError(
                "Unsupported inner underlying dataset for class `TorchvisionDataset` in `SubsetCache`"
            )
        return indices, subset

    def _get_random_torch_subsets(
        self, dataset: TorchVisionDataset
    ) -> tuple[TorchVisionDataset, TorchVisionDataset, float]:
        """Extract two random subsets from a torch dataset and compute their true
        Jaccard similarity"""
        indices_a, subset_a = self._get_random_torch_subset(dataset)
        indices_b, subset_b = self._get_random_torch_subset(dataset)

        jaccard = len(indices_a.intersection(indices_b)) / len(indices_a.union(indices_b))

        c_subset_a = TorchVisionDataset(subset_a)  # pyright: ignore[reportArgumentType]
        c_subset_b = TorchVisionDataset(subset_b)  # pyright: ignore[reportArgumentType]

        return c_subset_a, c_subset_b, jaccard

    def _get_random_arff_subsets(
        self, dataset: ARFFDataset
    ) -> tuple[ARFFDataset, ARFFDataset, float]:
        """Extract two random subsets from a csv dataset and compute their true
        Jaccard similarity"""
        indices_a = set(self._get_random_data_indices(dataset))
        indices_b = set(self._get_random_data_indices(dataset))
        jaccard = len(indices_a.intersection(indices_b)) / len(indices_a.union(indices_b))

        file_a = tempfile.NamedTemporaryFile(suffix=".arff").name
        file_b = tempfile.NamedTemporaryFile(suffix=".arff").name
        extract_arff_indices(dataset.arff_file, indices_a, file_a)
        extract_arff_indices(dataset.arff_file, indices_b, file_b)
        logger.debug("Extracted new ARFF subset in %s", file_a)
        logger.debug("Extracted new ARFF subset in %s", file_b)
        subset_a = ARFFDataset(file_a)
        subset_b = ARFFDataset(file_b)

        return subset_a, subset_b, jaccard


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
        dataset: TorchVisionDataset | ARFFDataset,
        algo: Algorithm,
        config: AccuracyConfig,
        subset_cache: Optional[SubsetCache] = None,
    ) -> None:
        self.dataset = dataset
        self.algo = algo
        self.config = config
        self.subset_cache = (
            subset_cache if subset_cache is not None else SubsetCache(AccuracyConfig())
        )

    def run(self) -> float:
        """Measure the accuracy of the fingerprint method. This function
        generates fingerprints for random subsets of the supplied dataset
        and compares their estimated similarity to the true similarity.

        It returns the average error over the number of tests. The lesser the better."""

        results: list[float] = []
        # Make sure we reset the cache so that we start from already
        # cached subsets before generating new ones.
        self.subset_cache.reset()
        for i in range(self.config.n_samples):
            logger.debug(f"Starting random accuracy test {i+1}/{self.config.n_samples}")
            results.append(self.one_random_accuracy_test())
        # Return average error
        return sum(results) / len(results)

    def one_random_accuracy_test(self) -> float:
        """Run a single random accuracy test.

        This returns the distance between estimated similarity and true
        similarity. In other words it returns the error in estimated
        similarity. The bigger the return value is the less accurate the
        estimated similarity is"""
        subset_a, subset_b, jaccard = self.subset_cache.get_subset_pair(self.dataset)
        algo_a = self.algo.clone_config()
        algo_b = self.algo.clone_config()
        algo_a.update(subset_a)
        algo_b.update(subset_b)
        fingerprint_a = algo_a.digest()
        fingerprint_b = algo_b.digest()
        assert isinstance(fingerprint_a, DatasetFingerprint) and isinstance(
            fingerprint_b, DatasetFingerprint
        )
        return abs(fingerprint_a.similarity(fingerprint_b) - jaccard)
