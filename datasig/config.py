from dataclasses import dataclass, field
import struct
from typing import List
import numpy as np


def gen_xor_magic_numbers():
    """Deterministically generate 400 random magic numbers"""
    # Use a fixed seed and separate rng to ensure each run of
    # the function returns the SAME magic numbers
    rng = np.random.default_rng(0)
    return [rng.bytes(32) for _ in range(400)]


@dataclass(frozen=True)
class ConfigV0:
    nb_signatures: int = 400
    lsh_magic_numbers: List[bytes] = field(
        default_factory=lambda: [struct.pack(">I", i) for i in list(range(1, 401))]
    )
    xor_magic_numbers: List[bytes] = field(
        default_factory=gen_xor_magic_numbers,
    )
