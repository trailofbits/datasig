from torchvision.datasets import MNIST
from datasig.dataset import (
    CanonicalDataset,
    Dataset,
    TorchVisionDataset,
    DatasetFingerprint,
)
from datasig.config import *
import numpy as np
import torch
from tqdm import tqdm
import typing as t
import argparse


class AccuracyRandomTester:
    def __init__(
        self, dataset: Dataset, tests: dict[str, AlgoV0], min_subset_size=1000
    ) -> None:
        self.dataset = dataset
        self.tests = tests
        self.subset_min_size = min_subset_size

    def _gen_subset(self) -> tuple[set[int], torch.utils.data.Subset]:
        subset_size = np.random.randint(self.subset_min_size, len(self.dataset))
        indices: np.ndarray = np.arange(len(self.dataset))
        np.random.shuffle(indices)

        return set(indices[:subset_size]), torch.utils.data.Subset(
            self.dataset, indices[:subset_size]
        )

    def run_random_test_case(self) -> dict[str, float]:
        indices_a, subset_a = self._gen_subset()
        indices_b, subset_b = self._gen_subset()

        jaccard = len(indices_a.intersection(indices_b)) / len(
            indices_a.union(indices_b)
        )

        c_subset_a = CanonicalDataset(TorchVisionDataset(subset_a))
        c_subset_b = CanonicalDataset(TorchVisionDataset(subset_b))

        return {
            name: abs(
                c_subset_a.gen_fingerprint(algo).similarity(
                    c_subset_b.gen_fingerprint(algo)
                )
                - jaccard
            )
            for name, algo in self.tests.items()
        }


def main(samples: int, min_subset_size: int) -> None:
    dataset = MNIST(root="/tmp/mnist_data", train=True, download=True)

    tests: dict[str, AlgoV0] = {
        "keyed_sha": KeyedSha(),
        "xor": Xor(),
        "single_sha": SingleSha(),
        "datasketch": Datasketch(),
    }
    mean_errs = {k: 0 for k in tests.keys()}

    tester = AccuracyRandomTester(dataset, tests, min_subset_size=min_subset_size)

    for i in tqdm(range(samples)):
        errs = tester.run_random_test_case()

        # Cumulative averages for all algos: C_{n+1} = C_n + \frac{1}{n+1} (x_n - C_n)
        mean_errs = {
            name: mean_err + (1 / (i + 1)) * (err - mean_err)
            for (name, err), mean_err in zip(errs.items(), mean_errs.values())
        }

    print(
        "Mean errors: "
        + ", ".join([f"{name}: {err}" for name, err in mean_errs.items()])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=20, type=int, help="Number of samples")
    parser.add_argument(
        "--min_subset_size", default=1000, type=int, help="Minimum subset size"
    )
    args = parser.parse_args()
    main(args.samples, args.min_subset_size)
