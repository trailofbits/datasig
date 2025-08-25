from collections.abc import Iterable
from abc import ABC, abstractmethod
import hashlib
import struct
import numpy as np
import datasketch  # pyright: ignore[reportMissingTypeStubs]
from .fingerprint import (
    DatasetFingerprint,
    DatasetUID,
    BasicDatasetFingerprint,
    DatasketchFingerprint,
)
from typing import Self


class Algorithm(ABC):
    @abstractmethod
    def update(self, data: bytes | Iterable[bytes]) -> None:
        pass

    @abstractmethod
    def digest(self) -> DatasetFingerprint | DatasetUID:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def clone_config(self) -> Self:
        pass


class MinHash(Algorithm, ABC):
    def __init__(self, data: Iterable[bytes]):
        self._hashes: list[bytes] = []
        self._synced: bool = True
        self._fingerprint: DatasetFingerprint | None = None

        for d in data:
            self.update(d)

    def _sort(self):
        self._hashes.sort()

    def update(self, data: bytes | Iterable[bytes]) -> None:
        self._synced = False
        if isinstance(data, bytes):
            self._hashes.append(hashlib.sha256(data).digest())
        else:
            self._hashes += [hashlib.sha256(x).digest() for x in data if isinstance(x, bytes)]

    @abstractmethod
    def _get_fingerprint(self) -> DatasetFingerprint:
        pass

    def digest(self) -> DatasetFingerprint:
        if not self._synced or self._fingerprint is None:
            if not self._synced:
                self._hashes.sort()
            self._fingerprint = self._get_fingerprint()

        return self._fingerprint


class KeyedShaMinHash(MinHash):
    """Generate dataset fingerprints using the keyed SHA method.

    The figerprint is computed using the MinHash scheme.
    Random permutations are approximated by keyed SHA256 functions i.e. random bytes are appended to the input of SHA256.

    See https://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html, Min Hash with k hash functions.
    """

    lsh_magic_numbers: list[bytes] = [struct.pack(">I", i) for i in list(range(1, 401))]

    def __init__(self, data: Iterable[bytes] = [], nb_signatures: int = 400):
        super().__init__(data)
        self.nb_signatures = nb_signatures

    def __str__(self) -> str:
        return f"KeyedShaMinHash(nb_signatures={self.nb_signatures})"

    def clone_config(self) -> Self:
        return self.__class__(nb_signatures=self.nb_signatures)

    def _get_fingerprint(self) -> DatasetFingerprint:
        if len(self._hashes) < self.nb_signatures:
            raise ValueError(
                f"Not enough data points to compute a fingerprint: we need at least {self.nb_signatures} data points."
            )

        res: list[bytes | None] = [None] * 400
        for i in range(self.nb_signatures):
            for h in self._hashes:
                h2 = hashlib.sha256(h + KeyedShaMinHash.lsh_magic_numbers[i]).digest()
                if res[i] is None or res[i] > h2:  # pyright: ignore[reportOptionalOperand]
                    res[i] = h2

        return BasicDatasetFingerprint(res)  # pyright: ignore[reportArgumentType]


def gen_xor_magic_numbers() -> list[bytes]:
    """Deterministically generate 400 random magic numbers"""
    # Use a fixed seed and separate rng to ensure each run of
    # the function returns the SAME magic numbers
    rng = np.random.default_rng(0)
    return [rng.bytes(32) for _ in range(400)]


class XorMinHash(MinHash):
    """Generate dataset fingerprint based on XOR permutations.

    The figerprint is computed using the MinHash scheme.
    Random permutations are approximated by the output of SHA256 XORed with random 256 bits values.
    """

    xor_magic_numbers: list[bytes] = gen_xor_magic_numbers()

    def __init__(self, data: Iterable[bytes] = [], nb_signatures: int = 400):
        super().__init__(data)
        self.nb_signatures = nb_signatures

    def __str__(self) -> str:
        return f"XorMinHash(nb_signatures={self.nb_signatures})"

    def clone_config(self) -> Self:
        return self.__class__(nb_signatures=self.nb_signatures)

    def _get_fingerprint(self) -> DatasetFingerprint:
        if len(self._hashes) < self.nb_signatures:
            raise ValueError(
                f"Not enough data points to compute a fingerprint: we need at least {self.nb_signatures} data points."
            )

        res: list[bytes | None] = [None] * self.nb_signatures
        for i in range(self.nb_signatures):
            for h in self._hashes:
                h2 = bytes([a ^ b for a, b in zip(h, XorMinHash.xor_magic_numbers[i])])
                if res[i] is None or res[i] > h2:  # pyright: ignore[reportOptionalOperand]
                    res[i] = h2

        return BasicDatasetFingerprint(res)  # pyright: ignore[reportArgumentType]


class SingleShaMinHash(MinHash):
    """Generate dataset fingerprint based on a single SHA permutation

    The figerprint is computed using the MinHash scheme with a single permutation approximated by SHA256.

    See https://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html, Min Hash with one hash functions.
    And https://en.wikipedia.org/wiki/MinHash#Variant_with_a_single_hash_function.
    """

    def __init__(self, data: Iterable[bytes] = [], nb_signatures: int = 400):
        super().__init__(data)
        self.nb_signatures = nb_signatures

    def __str__(self) -> str:
        return f"SingleShaMinHash(nb_signatures={self.nb_signatures})"

    def clone_config(self) -> Self:
        return self.__class__(nb_signatures=self.nb_signatures)

    def _get_fingerprint(self) -> DatasetFingerprint:
        # Relies on the fact that the data point hashes are sorted
        if len(self._hashes) < self.nb_signatures:
            raise ValueError(
                f"Not enough data points to compute a fingerprint: we need at least {self.nb_signatures} data points."
            )

        res = self._hashes[: self.nb_signatures]

        return BasicDatasetFingerprint(res)  # pyright: ignore[reportArgumentType]


class DatasketchMinHash(MinHash):
    """Generate dataset fingerprint based on universal hashing permutations

    The figerprint is computed using the MinHash scheme.
    Random permutations are approximated by universal hashing.

    See https://ekzhu.com/datasketch/minhash.html.
    We can customize this further using different hash functions and
    number of permutations.
    """

    def __init__(self, data: Iterable[bytes] = [], nb_signatures: int = 400):
        super().__init__(data)
        self.nb_signatures = nb_signatures

    def __str__(self) -> str:
        return f"DatasketchMinHash(nb_signatures={self.nb_signatures})"

    def clone_config(self) -> Self:
        return self.__class__(nb_signatures=self.nb_signatures)

    def _get_fingerprint(self) -> DatasetFingerprint:
        res = datasketch.MinHash(num_perm=self.nb_signatures)
        for d in self._hashes:
            res.update(d)  # pyright: ignore[reportUnknownMemberType]

        return DatasketchFingerprint(res)  # pyright: ignore[reportArgumentType]


class UID(Algorithm):
    """Generate dataset unique identifier"""

    def __init__(self, data: Iterable[bytes] = []) -> None:
        self._uid = hashlib.sha256()
        for h in data:
            self._uid.update(h)

    def __str__(self) -> str:
        return f"UID"

    def clone_config(self) -> Self:
        return self.__class__()

    def update(self, data: bytes | Iterable[bytes]) -> None:
        if isinstance(data, bytes):
            self._uid.update(data)
        else:
            for h in data:
                assert isinstance(h, bytes)
                self._uid.update(h)

    def digest(self) -> DatasetUID:
        # Hash the concatenation of the data points
        return DatasetUID(self._uid.digest())
