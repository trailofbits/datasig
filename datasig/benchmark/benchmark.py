from dataclasses import dataclass
from ..dataset import IterableDataset, TorchVisionDataset, ARFFDataset
from typing import Any
from ..algo import Algorithm
from .utils import catchtime
from .accuracy import AccuracyConfig, FingerprintAccuracyRandomTester, SubsetCache
import csv
from io import StringIO
from .plot import plot_results2
from .logger import logger
import uuid


@dataclass
class DatasetInfo:
    name: str
    format: str


# TODO(boyan): properly define the config type?
Config = Any


@dataclass
class BenchmarkResult:
    # Test case information
    dataset_info: DatasetInfo
    algorithm: str
    # config_name: str = ""
    # Time measurements in seconds
    time_canonization: float | None = None
    time_fingerprint: float | None = None
    # Accuracy measurements
    accuracy_error: float | None = None

    def __str__(self):
        res = ""
        res += f"Case: {self.algorithm} - {self.dataset_info.name}\n"
        if self.time_canonization is not None:
            res += f"  Time canonization: {self.time_canonization}\n"
        if self.time_fingerprint is not None:
            res += f"  Time fingerprint: {self.time_fingerprint}\n"
        if self.accuracy_error is not None:
            res += f"  Accuracy error: {self.accuracy_error}\n"
        return res


@dataclass
class BenchmarkResults:
    results: list[BenchmarkResult]

    def add(self, result: BenchmarkResult):
        """Add a result for one benchmark case"""
        self.results.append(result)

    def __str__(self):
        res = "## Benchmark results ##\n\n"
        for result in self.results:
            res += str(result) + "\n"
        return res

    def to_csv(self) -> str:
        res = StringIO()
        writer = csv.writer(res)
        writer.writerow(
            [
                "dataset",
                "algorithm",
                "config_name",
                "time_canonization",
                "time_fingerprint",
                "accuracy_error",
            ]
        )
        for result in self.results:
            writer.writerow(
                [
                    result.dataset_info.name,
                    result.algorithm,
                    result.time_canonization,
                    result.time_fingerprint,
                    result.accuracy_error,
                ]
            )
        return res.getvalue()

    def dump_csv(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_csv())

    # TODO(boyan): this plots the fingerprint time by default. Add other
    # metrics later
    def plot(self):
        return plot_results2(StringIO(self.to_csv()))


@dataclass
class BenchmarkConfig:
    # Configuration of the benchmark
    algos: list[Algorithm]
    # configs: Dict[str, Config]  # config_name -> config
    accuracy_config: AccuracyConfig | None = None
    # Which metrics to measure
    measure_time: bool = True


@dataclass
class Benchmark:
    name: str
    datasets: list[IterableDataset]
    config: BenchmarkConfig

    def run(self) -> BenchmarkResults:
        results = BenchmarkResults(results=[])
        # This cache is used to store subsets of datasets that used in the
        # benchmark and speed up the accuracy measurements by generating them
        # only once instead of generating them for each benchmark case.
        subsets_cache = (
            SubsetCache(self.config.accuracy_config)
            if self.config.accuracy_config is not None
            else None
        )
        # Temp file to store benchmark results as we go
        tmp_res_file = f"/tmp/{str(uuid.uuid4())[:8]}_datasig_benchmark.csv"
        # Benchmark each dataset
        for dataset in self.datasets:
            # With each algorithm and supplied config(s)
            for algo in self.config.algos:
                try:
                    logger.info(
                        "Running benchmark case: %s %s",
                        dataset.name,
                        str(algo),
                    )
                    res = BenchmarkResult(
                        dataset_info=DatasetInfo(dataset.name, str(type(dataset))),
                        algorithm=str(algo),
                    )
                    if self.config.measure_time:
                        logger.info("Start measuring fingerprinting time")
                        canonization_time, fingerprint_time = self._benchmark_fingerprint(
                            dataset, algo
                        )
                        res.time_canonization = canonization_time
                        res.time_fingerprint = fingerprint_time
                        logger.info("Done measuring fingerprinting time")
                    # Optionally measure fingerprint accuracy
                    if self.config.accuracy_config is not None and isinstance(
                        dataset, (TorchVisionDataset, ARFFDataset)
                    ):
                        logger.info("Start measuring fingerprint accuracy")
                        res.accuracy_error = FingerprintAccuracyRandomTester(
                            dataset, algo, self.config.accuracy_config, subsets_cache
                        ).run()
                        logger.info("Done measuring fingerprint accuracy")
                    results.add(res)
                    logger.info("Dumping temporary benchmark results to file %s", tmp_res_file)
                    results.dump_csv(tmp_res_file)
                except KeyboardInterrupt:
                    logger.warning("Benchmark case interrupted")
                    continue
        return results

    def _benchmark_fingerprint(
        self,
        dataset: IterableDataset,
        algo: Algorithm,
    ) -> tuple[float, float]:
        """Benchmark the time needed to generate a fingerprint for a dataset
        with a given algorithm"""
        # TODO(boyan): we could factorize the canonization step. Technically
        # we use the same canonical dataset for each config, and potentially
        # for each algorithm even. Canonizing only once could
        # help us save a lot of time when benchmarking on the same dataset
        # (which is the common case)
        with catchtime() as canonization_time:
            logger.debug("Canonizing dataset %s", dataset.name)
            algo.update(dataset)
        with catchtime() as fingerprint_time:
            logger.debug(
                "Fingerprinting dataset %s with algorithm %s",
                dataset.name,
                str(algo),
            )
            _ = algo.digest()
        return canonization_time(), fingerprint_time()
