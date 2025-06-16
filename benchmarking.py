from datasig.benchmark import (
    Benchmark,
    BenchmarkConfig,
    BASIC_FINGERPRINT,
    XOR_FINGERPRINT,
    SINGLE_SHA_FINGERPRINT,
    DATASKETCH_FINGERPRINT,
    AccuracyConfig,
)
from torchvision.datasets import MNIST
from datasig.config import ConfigV0
from datasig.dataset import TorchVisionDataset, ARFFDataset
from datasig.test.utils import download_file


# Example benchmark run on the MNIST dataset
def main():
    # Get datasets to benchmark on
    data = MNIST(root="/tmp/mnist_data", train=True, download=True)
    dataset_1 = TorchVisionDataset(data)

    arff_file = download_file(
        "https://www.openml.org/data/download/2169/BayesianNetworkGenerator_trains.arff",
        "/tmp/data.arff",
    )
    dataset_2 = ARFFDataset(arff_file)

    # Configure benchmark
    benchmark = Benchmark(
        name="Test benchmark on MNIST",
        # Datasets to benchmark on
        datasets=[
            dataset_1,
            dataset_2,
        ],
        config=BenchmarkConfig(
            # Fingerprint methods to benchmark
            methods=[
                # BASIC_FINGERPRINT,
                # XOR_FINGERPRINT,
                SINGLE_SHA_FINGERPRINT,
                DATASKETCH_FINGERPRINT,
            ],
            # Configs to benchmark with
            configs={"default": ConfigV0()},
            # How to test fingerprint accuracy. Only needed if measure_accuracy is True
            accuracy_config=AccuracyConfig(
                min_subset_size=1000,
                max_subset_size=3000,
                n_samples=5,
            ),
            # Whether to measure fingerprint generation time
            measure_time=True,
            # Whether to measure fingerprint accuracy
            measure_accuracy=True,
        ),
    )

    # Run the benchmark
    results = benchmark.run()
    print(results)
    results.plot()


if __name__ == "__main__":
    main()
