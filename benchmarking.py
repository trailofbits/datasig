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
from datasig.dataset import TorchVisionDataset

# Example benchmark run on the MNIST dataset
def main():
    # Get datasets to benchmark on
    data = MNIST(root="/tmp/mnist_data", train=True, download=True)
    dataset = TorchVisionDataset(data)

    # Configure benchmark
    benchmark = Benchmark(
        name="Test benchmark on MNIST",
        # Datasets to benchmark on
        datasets=[dataset],
        config=BenchmarkConfig(
            # Fingerprint methods to benchmark
            methods=[BASIC_FINGERPRINT, SINGLE_SHA_FINGERPRINT, DATASKETCH_FINGERPRINT],
            # Configs to benchmark with
            configs={"default": ConfigV0()},
            # How to test fingerprint accuracy. Only needed if measure_accuracy is True
            accuracy_config=AccuracyConfig(
                min_subset_size=1000,
                max_subset_size=3000,
                n_samples=10,
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


if __name__ == "__main__":
    main()
