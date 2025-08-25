from datasig.benchmark import (
    Benchmark,
    BenchmarkConfig,
    AccuracyConfig,
)
from datasig.algo import SingleShaMinHash, DatasketchMinHash
from torchvision.datasets import MNIST  # pyright: ignore[reportMissingTypeStubs]
from datasig.dataset import TorchVisionDataset, ARFFDataset
from datasig.test.utils import download_file


# Example benchmark run
def main():
    # Get datasets to benchmark on
    data = MNIST(root="/tmp/mnist_data", train=True, download=True)
    dataset_1 = TorchVisionDataset(data)  # pyright: ignore[reportArgumentType]

    arff_file = download_file(
        "https://www.openml.org/data/download/2169/BayesianNetworkGenerator_trains.arff",
        "/tmp/data.arff",
    )
    dataset_2 = ARFFDataset(arff_file)

    # Configure benchmark
    benchmark = Benchmark(
        name="Example benchmark",
        # Datasets to benchmark on
        datasets=[
            dataset_1,
            dataset_2,
        ],
        config=BenchmarkConfig(
            # Fingerprint methods to benchmark
            algos=[
                SingleShaMinHash(nb_signatures=400),
                DatasketchMinHash(nb_signatures=400),
            ],
            # How to test fingerprint accuracy. None to omit.
            accuracy_config=AccuracyConfig(
                min_subset_size=1000,
                max_subset_size=3000,
                n_samples=5,
            ),
            # Whether to measure fingerprint generation time
            measure_time=True,
        ),
    )

    # Run the benchmark
    results = benchmark.run()
    print(results)  # Pretty print results in terminal
    results.plot()  # Plot results in histogram format


if __name__ == "__main__":
    main()
