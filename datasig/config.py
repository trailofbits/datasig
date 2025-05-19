from dataclasses import dataclass, field
import struct
from typing import Any, List


@dataclass(frozen=True)
class ConfigV0:
    nb_signatures: int = 400
    lsh_magic_numbers: List[bytes] = field(
        default_factory=lambda: [struct.pack(">I", i) for i in list(range(1, 401))]
    )
