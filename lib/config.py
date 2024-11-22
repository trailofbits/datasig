from dataclasses import dataclass
import struct


@dataclass(frozen=True)
class ConfigV0:
    nb_signatures = 400
    lsh_magic_numbers = [struct.pack(">I", i) for i in list(range(1, 401))]
