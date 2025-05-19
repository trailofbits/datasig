from dataclasses import dataclass, asdict
from ..dataset import Dataset, DatasetType, CanonicalDataset
from typing import List, Dict, Callable, Optional, Any
from enum import StrEnum
from ..fingerprint import DatasetFingerprint
from .utils import catchtime

FingerprintMethod = Callable[[CanonicalDataset], DatasetFingerprint]


def BASIC_FINGERPRINT(dataset: CanonicalDataset) -> DatasetFingerprint:
    return dataset.fingerprint


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
    similarity_accuracy: Optional[float] = None

    def __str__(self):
        res = ""
        res += f"Case: {self.fingerprint_method} - {self.config_name} config - {self.dataset_info.name}\n"
        if self.time_canonization is not None:
            res += f"  Time canonization: {self.time_canonization}\n"
        if self.time_fingerprint is not None:
            res += f"  Time fingerprint: {self.time_fingerprint}\n"
        if self.similarity_accuracy is not None:
            res += f"  Similarity accuracy: {self.similarity_accuracy}\n"
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


# TODO(boyan): add whether to measure accuracy to or not??? Maybe
# in a separate file... (because diff than normal benchmark)
@dataclass
class BenchmarkConfig:
    methods: List[FingerprintMethod]
    configs: Dict[str, Config]  # config_name -> config


@dataclass
class Benchmark:
    name: str
    datasets: List[Dataset]
    config: BenchmarkConfig

    def run(self) -> BenchmarkResults:
        results = BenchmarkResults(results=[])
        # Benchmark each dataset
        for dataset in self.datasets:
            # With each fingerprint method and supplied config(s)
            for method in self.config.methods:
                for config_name, config in self.config.configs.items():
                    res = self._benchmark_fingerprint(dataset, method, config)
                    res.config_name = config_name
                    results.add(res)
        return results

    def _benchmark_fingerprint(
        self, dataset: Dataset, fingerprint_method: FingerprintMethod, config: Config
    ) -> BenchmarkResult:
        """Benchmark the time needed to generate a fingerprint for a dataset with a given
        fingerprint method and one or several configurations"""
        # TODO(boyan): we could factorize the canonization step. Technically
        # we use the same canonical dataset for each config, and potentially
        # for each fingerprint method even. Canonizing only once could
        # help us save a lot of time when benchmarking on the same dataset
        # (which is the common case)
        with catchtime() as canonization_time:
            c_dataset = CanonicalDataset(dataset, config=config)
        with catchtime() as fingerprint_time:
            fingerprint = fingerprint_method(c_dataset)
        return BenchmarkResult(
            dataset_info=DatasetInfo(dataset.name, dataset.type),
            fingerprint_method=fingerprint_method.__name__,
            time_canonization=canonization_time(),
            time_fingerprint=fingerprint_time(),
        )
