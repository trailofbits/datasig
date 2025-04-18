from dataclasses import dataclass
from typing import List
import math


class DatasetUID(bytes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self) != 32:
            raise ValueError("Dataset UID must be a valid SHA256 hash. Length is invalid.")
        if not all(c in string.hexdigits for c in self):
            raise ValueError("Dataset UID must be a valid SHA256 hash. Found an invalid character.")


@dataclass
class DatasetFingerprint:
    signatures: List[bytes]

    def __len__(self):
        return len(self.signatures)

    def similarity(self, other: "DatasetFingerprint"):
        assert len(self) == len(other)
        # Count how many signature pairs are made of identical signatures
        return sum(map(lambda x: x[0] == x[1], zip(self.signatures, other.signatures))) / len(self)

    def comparison_accuracy(self) -> float:
        """Return the accuracy that this fingerprint has when compared to another fingerprint
        to approximate the Jaccard distance between two canonical datasets"""
        if self.signatures:
            return 1.0 - (1.0 / math.sqrt(len(self.signatures)))
        return 0.0
