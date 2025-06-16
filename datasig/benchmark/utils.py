from time import process_time
from contextlib import contextmanager
from typing import Callable
from datasig.dataset import CanonicalDataset, DatasetFingerprint
import arff

# NOTE(boyan): taken from https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
@contextmanager
def catchtime() -> Callable[[], float]:
    # TODO(boyan): use thread_time?
    t1 = t2 = process_time()
    yield lambda: t2 - t1
    t2 = process_time()


# Method used to generate the fingerprint
FingerprintMethod = Callable[[CanonicalDataset], DatasetFingerprint]


# Wrap basic fingerprint generation
def BASIC_FINGERPRINT(dataset: CanonicalDataset) -> DatasetFingerprint:
    return dataset.fingerprint


def XOR_FINGERPRINT(dataset: CanonicalDataset) -> DatasetFingerprint:
    return dataset.xor_fingerprint


def SINGLE_SHA_FINGERPRINT(dataset: CanonicalDataset) -> DatasetFingerprint:
    return dataset.single_sha_fingerprint


def DATASKETCH_FINGERPRINT(dataset: CanonicalDataset) -> DatasetFingerprint:
    return dataset.datasketch_fingerprint


def extract_arff_indices(arff_file: "str", indices: set[int], outfile: str) -> str:
    """Extracts a range of data points from an ARFF file into another ARFF file"""
    with open(arff_file, "r") as f:
        with open(outfile, "w") as outf:
            data = arff.load(f)
            data["data"] = [x for i, x in enumerate(data["data"]) if i in indices]
            outf.write(arff.dumps(data))
    return outfile
