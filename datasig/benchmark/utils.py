from time import process_time
from contextlib import contextmanager
from typing import Callable
from datasig.dataset import CanonicalDataset, DatasetFingerprint


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
