from dataclasses import dataclass
from typing import List
import math
from abc import ABC, abstractmethod
import string


class DatasetUID(bytes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self) != 32:
            raise ValueError("Dataset UID must be a valid SHA256 hash. Length is invalid.")
        if not all(c in string.hexdigits for c in self):
            raise ValueError("Dataset UID must be a valid SHA256 hash. Found an invalid character.")


# Astract fingerprint class
class DatasetFingerprint(ABC):
    @abstractmethod
    def __len__(self):
        """Return information about the size of the fingerprint. The meaning
        of the size depends on the fingerprint method."""
        pass

    @abstractmethod
    def similarity(self, other: "DatasetFingerprint"):
        """Return the dataset similary estimate between the datasets that this
        fingerprint and the other fingerprint correspond to."""
        pass

    @abstractmethod
    def comparison_accuracy(self) -> float:
        """Return the accuracy that this fingerprint has when compared to another fingerprint
        to approximate the Jaccard distance between two canonical datasets"""
        pass


@dataclass
class BasicDatasetFingerprint(DatasetFingerprint):
    """A basic dataset fingerprint that stores the minhash signatures
    as a raw list of bytes"""

    signatures: List[bytes]

    def __len__(self):
        return len(self.signatures)

    def similarity(self, other: "BasicDatasetFingerprint"):
        assert len(self) == len(other)
        # Count how many signature pairs are made of identical signatures
        return sum(map(lambda x: x[0] == x[1], zip(self.signatures, other.signatures))) / len(self)

    def comparison_accuracy(self) -> float:
        if self.signatures:
            return 1.0 - (1.0 / math.sqrt(len(self.signatures)))
        return 0.0
