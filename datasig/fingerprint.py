from dataclasses import dataclass
from typing import List
import math
import string
from datasketch import MinHash
from enum import Enum, auto


class MinHashType(Enum):
    MULTI = auto()
    SINGLE = auto()
    DATASKETCH = auto()


class DatasetUID(bytes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self) != 32:
            raise ValueError(
                "Dataset UID must be a valid SHA256 hash. Length is invalid."
            )
        if not all(c in string.hexdigits for c in self):
            raise ValueError(
                "Dataset UID must be a valid SHA256 hash. Found an invalid character."
            )


@dataclass
class DatasetFingerprint:
    signatures: List[bytes] | MinHash
    min_hash_type: MinHashType

    def __len__(self):
        return len(self.signatures)

    def similarity(self, other: "DatasetFingerprint"):
        match self.min_hash_type:
            case MinHashType.DATASKETCH:
                return self.signatures.jaccard(other.signatures)
            case MinHashType.SINGLE:
                # Single permutation MinHash estimation: https://en.wikipedia.org/wiki/MinHash#Variant_with_a_single_hash_function
                ha = set(self.signatures)
                hb = set(other.signatures)

                u = list(ha.union(hb))
                list.sort(u)
                x = set(u[:400])

                y = x.intersection(ha, hb)

                return len(y) / 400
            case MinHashType.MULTI:
                # Count how many signature pairs are made of identical signatures
                assert len(self) == len(other)
                return sum(
                    map(lambda x: x[0] == x[1], zip(self.signatures, other.signatures))
                ) / len(self)

    def comparison_accuracy(self) -> float:
        """Return the accuracy that this fingerprint has when compared to another fingerprint
        to approximate the Jaccard distance between two canonical datasets"""
        if self.signatures:
            return 1.0 - (1.0 / math.sqrt(len(self.signatures)))
        return 0.0
