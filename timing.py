from torchvision.datasets import MNIST
from datasig.dataset import CanonicalDataset, Dataset, TorchVisionDataset
from datasig.config import *
from tqdm import tqdm
from time import thread_time
import argparse


class TimeTester:
    def __init__(
        self,
        dataset: Dataset,
        tests: dict[str, AlgoV0],
        canonical: bool,
        fingerprint: bool,
    ) -> None:
        self.tv_dataset = TorchVisionDataset(dataset)
        self.tests = tests
        self.canonical = canonical
        self.fingerprint = fingerprint

    def run_test_case(self) -> dict[str, float]:
        res = {}

        for name, algo in self.tests.items():
            c_start = thread_time() if self.canonical else 0
            c_dataset = CanonicalDataset(self.tv_dataset)
            c_time = thread_time() - c_start if self.canonical else 0

            f_start = thread_time() if self.fingerprint else 0
            if self.fingerprint:
                _ = c_dataset.gen_fingerprint(algo)
            f_time = thread_time() - f_start if self.fingerprint else 0

            res[name] = c_time + f_time

        return res


def main(samples: int, canonical: bool, fingerprint: bool) -> None:
    dataset = MNIST(root="/tmp/mnist_data", train=True, download=True)

    tests: dict[str, AlgoV0] = {
        "keyed_sha": KeyedSha(),
        "xor": Xor(),
        "single_sha": SingleSha(),
        "datasketch": Datasketch(),
    }
    mean_times = {k: 0.0 for k in tests.keys()}

    tester = TimeTester(dataset, tests, canonical, fingerprint)

    for i in tqdm(range(samples)):
        iter_times = tester.run_test_case()

        # Cumulative averages for all algos: C_{n+1} = C_n + \frac{1}{n+1} (x_n - C_n)
        mean_times = {
            name: mean_time + (1 / (i + 1)) * (iter_time - mean_time)
            for (name, iter_time), mean_time in zip(
                iter_times.items(), mean_times.values()
            )
        }

    print(
        "Mean times: "
        + ", ".join([f"{name}: {time}" for name, time in mean_times.items()])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=20, type=int, help="Number of samples")
    parser.add_argument(
        "--canonical", action="store_true", help="Include dataset canonization in stats"
    )
    parser.add_argument(
        "--fingerprint",
        action="store_true",
        help="Include fingerprint computation in stats",
    )
    args = parser.parse_args()
    main(args.samples, args.canonical, args.fingerprint)
