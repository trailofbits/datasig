from dataclasses import dataclass
import struct
import numpy as np


@dataclass(frozen=True)
class ConfigV0:
    nb_signatures = 400
    lsh_magic_numbers = [struct.pack(">I", i) for i in list(range(1, 401))]
    xor_numbers = [np.random.bytes(32) for _ in range(400)]
