from datasig.benchmark import Benchmark, BenchmarkConfig, BASIC_FINGERPRINT
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
            methods=[BASIC_FINGERPRINT],
            # Configs to benchmark with
            configs={"default": ConfigV0()},
        ),
    )

    # Run the benchmark
    results = benchmark.run()
    print(results)


if __name__ == "__main__":
    main()
