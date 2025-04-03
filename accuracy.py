from torchvision.datasets import MNIST
from datasig.dataset import CanonicalDataset, Dataset, TorchVisionDataset, DatasetFingerprint
import numpy as np
import torch
from tqdm import tqdm
import typing as t


class RandomTester:
    def __init__(self, dataset: Dataset, tests: dict[str, t.Callable[[CanonicalDataset], DatasetFingerprint]]) -> None:
        self.tests = tests
        c_dataset = CanonicalDataset(TorchVisionDataset(dataset))

        self.fingerprints = { name: f(c_dataset) for name, f in tests.items() }
        # self.merc_fingerprint = DatasketchDataset(tv_dataset)._fingerprint

    def run_random_test_case(self) -> dict[str, float]:
        subset_size = np.random.randint(1000, len(dataset))

        indices: np.ndarray = np.arange(len(dataset))
        np.random.shuffle(indices)
        subset = torch.utils.data.Subset(dataset, indices[:subset_size])

        assert subset_size == len(subset)
        jaccard = subset_size / len(dataset)

        c_subset = CanonicalDataset(TorchVisionDataset(subset))

        return { name: abs(self.fingerprints[name].similarity(f(c_subset)) - jaccard) for name, f in self.tests.items() }


if __name__ == "__main__":
    dataset = MNIST(root="/tmp/mnist_data", train=True, download=True)

    tests: dict[str, t.Callable[[CanonicalDataset], DatasetFingerprint]] = {
        "keyed_sha": lambda x: x.fingerprint,
        "xor": lambda x: x.xor_fingerprint,
        "single_sha": lambda x: x.single_fingerprint,
        "datasketch": lambda x: x.datasketch_fingerprint,
    }
    mean_errs = { k: 0 for k in tests.keys() }

    tester = RandomTester(dataset, tests)

    for i in tqdm(range(20)):
        errs = tester.run_random_test_case()
        mean_errs = { name: mean_err + (1 / (i + 1)) * (err - mean_err) for (name, err), mean_err in zip(errs.items(), mean_errs.values()) }
    
    print("Mean errors: " + ", ".join([f"{name}: {err}" for name, err in mean_errs.items()]))
