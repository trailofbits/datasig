from time import process_time
from contextlib import contextmanager
from typing import Generator, Callable, Any
import arff  # pyright: ignore[reportMissingTypeStubs]
from pathlib import Path


# NOTE(boyan): taken from https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
@contextmanager
def catchtime() -> Generator[Callable[[], float]]:
    # TODO(boyan): use thread_time?
    t1 = t2 = process_time()
    yield lambda: t2 - t1
    t2 = process_time()


# Method used to generate the fingerprint
# FingerprintMethod = Callable[[CanonicalDataset, Any], DatasetFingerprint]


# Wrap basic fingerprint generation
# def BASIC_FINGERPRINT(dataset: CanonicalDataset, config: Any) -> DatasetFingerprint:
#     return dataset._fingerprint(config)


# def XOR_FINGERPRINT(dataset: CanonicalDataset, config: Any) -> DatasetFingerprint:
#     return dataset._xor_fingerprint(config)


# def SINGLE_SHA_FINGERPRINT(dataset: CanonicalDataset, config: Any) -> DatasetFingerprint:
#     return dataset._single_sha_fingerprint(config)


# def DATASKETCH_FINGERPRINT(dataset: CanonicalDataset, config: Any) -> DatasetFingerprint:
#     return dataset._datasketch_fingerprint(config)


def extract_arff_indices(arff_file: Path | str, indices: set[int], outfile: str) -> str:
    """Extracts a range of data points from an ARFF file into another ARFF file"""
    with open(arff_file, "r") as f:
        with open(outfile, "w") as outf:
            data: dict[str, Any] = arff.load(
                f
            )  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            data["data"] = [x for i, x in enumerate(data["data"]) if i in indices]
            outf.write(arff.dumps(data))  # pyright: ignore[reportUnknownMemberType]
    return outfile
