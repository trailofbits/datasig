import math
from abc import ABC, abstractmethod
import datasketch  # pyright: ignore[reportMissingTypeStubs]


class DatasetUID(bytes):
    def __new__(cls, data: bytes = b""):
        if len(data) != 32:
            raise ValueError("Dataset UID must be a valid SHA256 hash. Length is invalid.")

        return super().__new__(cls, data)


# Astract fingerprint class
class DatasetFingerprint(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """Return information about the size of the fingerprint. The meaning
        of the size depends on the fingerprint method."""
        pass

    @abstractmethod
    def similarity(self, other: "DatasetFingerprint") -> float:
        """Return the dataset similary estimate between the datasets that this
        fingerprint and the other fingerprint correspond to."""
        pass

    @abstractmethod
    def comparison_accuracy(self) -> float:
        """Return the accuracy that this fingerprint has when compared to another fingerprint
        to approximate the Jaccard distance between two canonical datasets"""
        pass

    @abstractmethod
    def signature(self) -> bytes:
        pass


class BasicDatasetFingerprint(DatasetFingerprint):
    """A basic dataset fingerprint that stores the minhash signatures
    as a raw list of bytes"""

    def __init__(self, signatures: list[bytes]) -> None:
        self._signatures = signatures

    def __len__(self) -> int:
        return len(self._signatures)

    def similarity(self, other: "DatasetFingerprint") -> float:
        assert len(self) == len(other)
        assert isinstance(other, BasicDatasetFingerprint)
        # Count how many signature pairs are made of identical signatures
        return sum(map(lambda x: x[0] == x[1], zip(self._signatures, other._signatures))) / len(
            self
        )

    def comparison_accuracy(self) -> float:
        if self._signatures:
            return 1.0 - (1.0 / math.sqrt(len(self._signatures)))
        return 0.0

    def signature(self) -> bytes:
        return b"".join(self._signatures)


class DatasketchFingerprint(DatasetFingerprint):
    """A fingerprint that stores a minhash object from the
    datasketch library"""

    def __init__(self, minhash: datasketch.MinHash):
        self.minhash = minhash

    def __len__(self):
        return len(self.minhash)

    def similarity(self, other: "DatasetFingerprint"):
        assert isinstance(other, DatasketchFingerprint)
        return self.minhash.jaccard(other.minhash)

    def comparison_accuracy(self) -> float:
        # TODO(boyan): make sure using the length of the minhash
        # is correct for estimating the accuracy here
        l = len(self.minhash)
        if l:
            return 1.0 - (1.0 / math.sqrt(l))
        return 0.0

    def signature(self) -> bytes:
        return self.minhash.digest().tobytes()
