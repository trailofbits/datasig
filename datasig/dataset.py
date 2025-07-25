from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Generator, Union, Iterable, Dict, Optional
import hashlib
from .fingerprint import (
    DatasetUID,
    DatasketchFingerprint,
    DatasetFingerprint,
    BasicDatasetFingerprint,
)
from .config import ConfigV0
import torch
from PIL import Image
from pathlib import Path
import io
import csv
from abc import ABC, abstractmethod
import datasketch
import json
import base64


class DatasetType(Enum):
    SIMPLE_FILE_TREE = "File based"
    TORCH = "Torch dataset format"
    CSV = "CSV file"
    ARFF = "ARFF file"
    RAW = "List of raw bytes data points"


class Dataset(ABC):
    """An abstract class that represents a dataset to be fingerprinted"""

    def __init__(self, t: DatasetType):
        self.type = t

    @property
    def encoded_data_points(self) -> Generator[bytes, None, None]:
        """Iterate through all data points (transformed as bytes)"""
        for x in self.data_points:
            yield self.encode_data_point(x)

    @property
    @abstractmethod
    def data_points(self) -> Generator[Any, None, None]:
        """Iterate through all data points"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifying the dataset"""
        pass

    @abstractmethod
    def encode_data_point(self, data_point: Any) -> bytes:
        """
        Encode a data point into bytes for it to be fingerprinted.
        For transmitting serialized data points over the network, use
        the `serialized_data_point` interface instead
        """
        pass

    @staticmethod
    @abstractmethod
    def serialize_data_point(data: Any) -> bytes:
        """
        Serialize a data point into bytes for it to be transmitted over
        a network interface
        """
        pass

    @staticmethod
    @abstractmethod
    def deserialize_data_point(data: bytes) -> Any:
        """
        Instantiate a data point from bytes that were generated using
        the `serialize_data_point` method
        """
        pass


# TODO(boyan): this should be versioned depending on the torchvision version as they might change their
# underlying format
# TODO(boyan): see whether we can use this class for torch datasets in general, or just torchvision.
class TorchVisionDataset(Dataset):
    """Dataset loaded using the torch python API"""

    def __init__(self, dataset: torch.utils.data.TensorDataset):
        super().__init__(DatasetType.TORCH)
        self.dataset = dataset

    def __len__(self) -> int:
        """Return the number of data points in the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Image.Image:
        return self.dataset[idx][0]

    @property
    def name(self) -> str:
        """Name identifying the dataset"""
        # In torch the dataset class names are descriptive like `MNIST`
        return type(self.dataset).__name__

    @property
    def data_points(self) -> Generator[Image.Image, None, None]:
        """Iterate through all data points (transformed as bytes)"""
        for x, _ in self.dataset:
            yield x

    def encode_data_point(self, data: Image.Image) -> bytes:
        """Tranform a data point to its `bytes` representation"""
        if isinstance(data, Image.Image):
            return self._PIL_image_to_bytes_v0(data)
        else:
            raise Exception(f"Unsupported data point type: {type(data)}")

    @staticmethod
    def _PIL_image_to_bytes_v0(data: Image.Image) -> bytes:
        # Return raw image data
        return data.tobytes(encoder_name="raw")

    @staticmethod
    def serialize_data_point(data: Image) -> bytes:
        return json.dumps(
            [
                data.mode,
                data.size,
                base64.b64encode(TorchVisionDataset._PIL_image_to_bytes_v0(data)).decode(),
            ]
        )

    @staticmethod
    def deserialize_data_point(data: bytes) -> Image.Image:
        mode, size, raw_bytes = json.loads(data)
        return Image.frombytes(mode, size, base64.b64decode(raw_bytes.encode()), decoder_name="raw")


class ARFFDataset(Dataset):
    """Dataset contained in an ARFF file"""

    def __init__(self, arff_file: Union[Path, str]):
        super().__init__(DatasetType.ARFF)
        self.arff_file = arff_file

    def __len__(self) -> int:
        """Return the number of data points in the dataset"""
        return sum(1 for _ in self.data_points)

    def __getitem__(self, idx: int) -> bytes:
        raise NotImplemented

    @property
    def name(self) -> str:
        """Name identifying the dataset"""
        return str(self.arff_file)

    @property
    def data_points(self) -> Generator[bytes, None, None]:
        """Iterate through all data points (transformed as bytes)"""
        # The parsing is based on https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
        with open(self.arff_file, "rb") as f:
            reached_data = False
            for line in f:
                if reached_data:
                    yield line
                reached_data |= line.startswith((b"@data", b"@DATA"))

    def encode_data_point(self, data_point: bytes) -> bytes:
        """Data points are already in bytes so keep them as is"""
        if isinstance(data_point, bytes):
            return data_point
        else:
            raise Exception(f"Unsupported data point type: {type(data)}")

    @staticmethod
    def serialize_data_point(data: bytes) -> bytes:
        # Data is already bytes
        return data

    @staticmethod
    def deserialize_data_point(data: bytes) -> bytes:
        return data


class CSVDataset(Dataset):
    """Dataset contained in a CSV file"""

    def __init__(
        self, csv_data: Union[Iterable[str], List[str], Path, str], dialect="excel", **fmtparams
    ):
        super().__init__(DatasetType.CSV)
        self.dialect = dialect
        self.fmtparams = fmtparams
        if isinstance(
            csv_data,
            (
                Path,
                str,
            ),
        ):
            # Read from static file
            self.csv_data = open(csv_data, "r", newline="\n")
            self._name = str(csv_data)
        else:
            # Already a list or iterable of rows
            self.csv_data = csv_data
            # TODO(boyan): proper naming here needs to come from the user
            self._name = "Raw CSV data supplied from python"
        self.csv_reader = csv.reader(self.csv_data, dialect=dialect, **fmtparams)
        # Used as a cache in case __getitem__ is called to
        # access data points by index.
        self.cached_rows = []

    def __del__(self):
        if isinstance(self.csv_data, io.IOBase):
            # If data came from a file we opened, close the file
            self.csv_data.close()

    def __getitem__(self, idx: int) -> List[str]:
        """
        This allows to access individual rows of the csv dataset.
        Note that it will result in the entire dataset being stored in
        memory.

        WARNING: should not be used in conjunction with `data_points`
        property as they both consume the csv_reader iterator!
        """
        # Iterate the csv reader until we get to the index
        remaining_steps = idx + 1 - len(self.cached_rows)
        for i in range(0, remaining_steps):
            self.cached_rows = next(self.csv_reader)

    @property
    def name(self) -> str:
        """Name identifying the dataset"""
        return self._name

    @property
    def data_points(self) -> Generator[List[str], None, None]:
        """Iterate through all data points (transformed as bytes)"""
        # FIXME(boyan): we must ignore the row that contains
        # the labels. The easiest way is to ignore the first row, but we
        # must be sure it's the one containing labels, if there are labels...
        for row in self.csv_reader:
            # NOTE(boyan): on very huge datasets we might disable caching?
            self.cached_rows.append(row)
            yield row

    def encode_data_point(self, row: List[str]) -> bytes:
        """Tranform a data point to its `bytes` representation"""
        res = io.StringIO()
        writer = csv.writer(res, dialect=self.csv_reader.dialect)
        writer.writerow(row)
        return bytes(res.getvalue(), "utf-8")

    @staticmethod
    def serialize_data_point(data: List[str]) -> bytes:
        return json.dumps(data)

    @staticmethod
    def deserialize_data_point(data: bytes) -> List[str]:
        return json.loads(data)


class RawDataset(Dataset):
    """Dataset from raw data points"""

    def __init__(self, name: str, raw_data: Optional[List[bytes]] = None):
        super().__init__(DatasetType.RAW)
        self.raw_data = [] if raw_data is None else raw_data
        self._name = name

    def __len__(self) -> int:
        """Return the number of data points in the dataset"""
        return len(self.dataset)

    def add_data_point(self, raw_data_point: bytes):
        """Add a data point by supplying the data point raw bytes"""
        self.raw_data.append(raw_data_point)

    @property
    def name(self) -> str:
        """Name identifying the dataset"""
        return self._name

    @property
    def data_points(self) -> Generator[bytes, None, None]:
        """Iterate through all data points (transformed as bytes)"""
        return iter(self.raw_data)

    def encode_data_point(self, data_point: bytes) -> bytes:
        # Already encoded in bytes
        return data_point

    @staticmethod
    def serialize_data_point(data: bytes) -> bytes:
        # Data is already bytes
        return data

    @staticmethod
    def deserialize_data_point(data: bytes) -> bytes:
        return data


class CanonicalDataset:
    """A format agnostic canonical dataset representation to compute fingerprints"""

    def __init__(self, dataset: Dataset, config=None):
        self.data_point_hashes: List[bytes] = []
        self._uid: Optional[DatasetUID] = None
        # Fingerprints computed with different methods
        self._fingerprints: Dict[str, DatasetFingerprint] = dict()
        self.preprocessed: bool = False
        self.default_config = config if config else ConfigV0()

        for d in dataset.encoded_data_points:
            self.add_data_point(d)

    def add_data_point(self, data: bytes) -> None:
        """Compute the hash of the data point and add it to the list."""
        h = hashlib.sha256(data).digest()
        self.data_point_hashes.append(h)

    def _preprocess(self):
        """Preprocessing steps required before computing fingerprint"""
        # Sort the hash list
        list.sort(self.data_point_hashes)
        self.preprocessed = True

    def _check_preprocess(self, raise_exc=False):
        if not self.preprocessed:
            if raise_exc:
                raise Exception("Canonical dataset must be preprocessed")
            else:
                self._preprocess()

    @property
    def uid(self) -> DatasetUID:
        """Generate dataset unique identifier"""
        self._check_preprocess()
        if self._uid is None:
            # Hash the concatenation of the data points
            hash_function = hashlib.sha256()
            for h in self.data_point_hashes:
                hash_function.update(h)
            self._uid = hash_function.digest()
        return self._uid

    @property
    def fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint using the default configuration"""
        return self._fingerprint()

    def _fingerprint(self, config=None) -> DatasetFingerprint:
        """Generate dataset fingerprint using the keyed SHA method.

        The figerprint is computed using the MinHash scheme.
        Random permutations are approximated by keyed SHA256 functions i.e. random bytes are appended to the input of SHA256.

        See https://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html, Min Hash with k hash functions.
        """
        self._check_preprocess()
        FP_KEY = "KEYED_SHA"
        if config is None:
            config = self.default_config
        if FP_KEY not in self._fingerprints:
            res = [None] * 400
            for i in range(config.nb_signatures):
                for h in self.data_point_hashes:
                    h2 = hashlib.sha256(h + config.lsh_magic_numbers[i]).digest()
                    if res[i] is None or res[i] > h2:
                        res[i] = h2
            self._fingerprints[FP_KEY] = BasicDatasetFingerprint(res)
        return self._fingerprints[FP_KEY]

    @property
    def xor_fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint using the default configuration"""
        return self._xor_fingerprint()

    def _xor_fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint based on XOR permutations.

        The figerprint is computed using the MinHash scheme.
        Random permutations are approximated by the output of SHA256 XORed with random 256 bits values.
        """
        self._check_preprocess()
        FP_KEY = "XOR"
        if config is None:
            config = self.default_config
        if FP_KEY not in self._fingerprints:
            res = [None] * 400
            for i in range(config.nb_signatures):
                for h in self.data_point_hashes:
                    # Bitwise XOR with the magic number
                    h2 = [a ^ b for a, b in zip(h, config.xor_magic_numbers[i])]
                    if res[i] is None or res[i] > h2:
                        res[i] = h2
            res = [bytes(x) for x in res]
            self._fingerprints[FP_KEY] = BasicDatasetFingerprint(res)
        return self._fingerprints[FP_KEY]

    @property
    def single_sha_fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint using the default configuration"""
        return self._single_sha_fingerprint()

    def _single_sha_fingerprint(self, config=None) -> DatasetFingerprint:
        """Generate dataset fingerprint based on a single SHA permutation

        The figerprint is computed using the MinHash scheme with a single permutation approximated by SHA256.

        See https://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html, Min Hash with one hash functions.
        And https://en.wikipedia.org/wiki/MinHash#Variant_with_a_single_hash_function.
        """
        self._check_preprocess()
        FP_KEY = "SINGLE_SHA"
        if config is None:
            config = self.default_config
        if FP_KEY not in self._fingerprints:
            # Relies on the fact that the data point hashes are sorted
            res = self.data_point_hashes[: config.nb_signatures]
            self._fingerprints[FP_KEY] = BasicDatasetFingerprint(res)
        return self._fingerprints[FP_KEY]

    @property
    def datasketch_fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint using the default configuration"""
        return self._datasketch_fingerprint()

    def _datasketch_fingerprint(self, config=None) -> DatasetFingerprint:
        """Generate dataset fingerprint based on universal hashing permutations

        The figerprint is computed using the MinHash scheme.
        Random permutations are approximated by universal hashing.

        See https://ekzhu.com/datasketch/minhash.html.
        We can customize this further using different hash functions and
        number of permutations.
        """
        self._check_preprocess()
        FP_KEY = "DATASKETCH"
        if config is None:
            config = self.default_config
        if FP_KEY not in self._fingerprints:
            res = datasketch.MinHash(num_perm=config.nb_signatures)
            for d in self.data_point_hashes:
                res.update(d)
            self._fingerprints[FP_KEY] = DatasketchFingerprint(res)
        return self._fingerprints[FP_KEY]
