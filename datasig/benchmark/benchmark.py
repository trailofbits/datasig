from dataclasses import dataclass, asdict
from ..dataset import Dataset, DatasetType, CanonicalDataset
from typing import List, Dict, Callable, Optional, Any
from enum import StrEnum
from ..fingerprint import DatasetFingerprint
from .utils import catchtime, FingerprintMethod
from .accuracy import AccuracyConfig, FingerprintAccuracyRandomTester
import csv
from typing import TextIO
from io import StringIO
from .plot import plot_results, plot_results2
from ..logger import logger
import uuid


@dataclass
class DatasetInfo:
    name: str
    format: DatasetType


# TODO(boyan): properly define the config type?
Config = Any


@dataclass
class BenchmarkResult:
    # Test case information
    dataset_info: DatasetInfo
    fingerprint_method: str
    config_name: str = ""
    # Time measurements in seconds
    time_canonization: Optional[float] = None
    time_fingerprint: Optional[float] = None
    # Accuracy measurements
    accuracy_error: Optional[float] = None

    def __str__(self):
        res = ""
        res += f"Case: {self.fingerprint_method} - {self.config_name} config - {self.dataset_info.name}\n"
        if self.time_canonization is not None:
            res += f"  Time canonization: {self.time_canonization}\n"
        if self.time_fingerprint is not None:
            res += f"  Time fingerprint: {self.time_fingerprint}\n"
        if self.accuracy_error is not None:
            res += f"  Accuracy error: {self.accuracy_error}\n"
        return res


@dataclass
class BenchmarkResults:
    results: List[BenchmarkResult]

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
                "fingerprint_method",
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
                    result.fingerprint_method,
                    result.config_name,
                    result.time_canonization,
                    result.time_fingerprint,
                    result.accuracy_error,
                ]
            )
        print(res.getvalue())
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
    methods: List[FingerprintMethod]
    configs: Dict[str, Config]  # config_name -> config
    accuracy_config: Optional[AccuracyConfig] = None  # used only if measure_accuracy is True
    # Which metrics to measure
    measure_time: bool = True
    measure_accuracy: bool = False


@dataclass
class Benchmark:
    name: str
    datasets: List[Dataset]
    config: BenchmarkConfig

    def run(self) -> BenchmarkResults:
        results = BenchmarkResults(results=[])
        # Temp file to store benchmark results as we go
        tmp_res_file = f"/tmp/{str(uuid.uuid4())[:8]}_datasig_benchmark.csv"
        # Benchmark each dataset
        for dataset in self.datasets:
            # With each fingerprint method and supplied config(s)
            for method in self.config.methods:
                for config_name, config in self.config.configs.items():
                    try:
                        logger.info(
                            "Running benchmark case: %s %s %s",
                            dataset.name,
                            method.__name__,
                            config_name,
                        )
                        res = BenchmarkResult(
                            dataset_info=DatasetInfo(dataset.name, dataset.type),
                            fingerprint_method=method.__name__,
                            config_name=config_name,
                        )
                        if self.config.measure_time:
                            logger.info("Start measuring fingerprinting time")
                            canonization_time, fingerprint_time = self._benchmark_fingerprint(
                                dataset, method, config
                            )
                            res.time_canonization = canonization_time
                            res.time_fingerprint = fingerprint_time
                            logger.info("Done measuring fingerprinting time")
                        # Optionally measure fingerprint accuracy
                        if self.config.measure_accuracy:
                            logger.info("Start measuring fingerprint accuracy")
                            res.accuracy_error = FingerprintAccuracyRandomTester(
                                dataset, method, config, self.config.accuracy_config
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
        self, dataset: Dataset, fingerprint_method: FingerprintMethod, config: Config
    ) -> tuple[float, float]:
        """Benchmark the time needed to generate a fingerprint for a dataset with a given
        fingerprint method and one configuration"""
        # TODO(boyan): we could factorize the canonization step. Technically
        # we use the same canonical dataset for each config, and potentially
        # for each fingerprint method even. Canonizing only once could
        # help us save a lot of time when benchmarking on the same dataset
        # (which is the common case)
        with catchtime() as canonization_time:
            logger.debug("Canonizing dataset %s", dataset.name)
            c_dataset = CanonicalDataset(dataset, config=config)
        with catchtime() as fingerprint_time:
            logger.debug(
                "Fingerprinting dataset %s with method %s",
                dataset.name,
                fingerprint_method.__name__,
            )
            fingerprint = fingerprint_method(c_dataset)
        return canonization_time(), fingerprint_time()
