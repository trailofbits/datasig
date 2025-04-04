from dataclasses import dataclass, field
import struct
import numpy as np


KEYED_SHA_DEFAULT_NB_SIGNATURES = 400
XOR_DEFAULT_NB_SIGNATURES = 400
SINGLE_SHA_DEFAULT_NB_SIGNATURES = 400


@dataclass()
class KeyedSha:
    """Dataset fingerprints based on keyed-SHA permutations.

    The figerprint is computed using the MinHash scheme.
    Random permutations are approximated by keyed SHA256 functions i.e. random bytes are appended to the input of SHA256.

    See https://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html, Min Hash with k hash functions.
    """

    nb_signatures: int = KEYED_SHA_DEFAULT_NB_SIGNATURES
    magic_numbers: list[bytes] = field(
        default_factory=lambda: [
            struct.pack(">I", i)
            for i in list(range(1, KEYED_SHA_DEFAULT_NB_SIGNATURES + 1))
        ]
    )


@dataclass(frozen=True)
class Xor:
    """Dataset fingerprints based on XOR permutations.

    The figerprint is computed using the MinHash scheme.
    Random permutations are approximated by the output of SHA256 XORed with random 256 bits values.
    """

    nb_signatures: int = XOR_DEFAULT_NB_SIGNATURES
    magic_numbers: list[bytes] = field(
        default_factory=lambda: [
            np.random.bytes(32) for _ in range(XOR_DEFAULT_NB_SIGNATURES)
        ]
    )


@dataclass(frozen=True)
class SingleSha:
    """Dataset fingerprints based on a single SHA permutation

    The figerprint is computed using the MinHash scheme with a single permutation approximated by SHA256.

    See https://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html, Min Hash with one hash functions.
    And https://en.wikipedia.org/wiki/MinHash#Variant_with_a_single_hash_function.
    """

    nb_signatures: int = SINGLE_SHA_DEFAULT_NB_SIGNATURES


@dataclass(frozen=True)
class Datasketch:
    """Dataset fingerprints based on universal hashing permutations

    The figerprint is computed using the MinHash scheme.
    Random permutations are approximated by universal hashing.

    See https://ekzhu.com/datasketch/minhash.html.
    """


AlgoV0 = KeyedSha | Xor | SingleSha | Datasketch


__all__ = ["AlgoV0", "KeyedSha", "Xor", "SingleSha", "Datasketch"]
