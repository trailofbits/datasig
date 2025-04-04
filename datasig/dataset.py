from enum import Enum
import typing as t
import hashlib
from .fingerprint import DatasetUID, DatasetFingerprint, MinHashType
from .config import *
import torch
import PIL
from pathlib import Path
import io
import csv
from datasketch import MinHash


class DatasetType(Enum):
    SIMPLE_FILE_TREE = "File based"
    TORCH = "Torch dataset format"
    CSV = "CSV file"
    ARFF = "ARFF file"


class Dataset:
    """An abstract class that represents a dataset to be fingerprinted"""

    def __init__(self, t: DatasetType):
        self.type = t


# TODO(boyan): this should be versioned depending on the torchvision version as they might change their
# underlying format
# TODO(boyan): see whether we can use this class for torch datasets in general, or just torchvision.
class TorchVisionDataset(Dataset):
    """Dataset loaded using the torch python API"""

    def __init__(self, dataset: torch.utils.data.TensorDataset):
        super().__init__(DatasetType.TORCH)
        self.dataset = dataset

    @property
    def data_points(self) -> t.Generator[bytes, None, None]:
        """Iterate through all data points (transformed as bytes)"""
        for x, _ in self.dataset:
            yield self.data_point_to_bytes(x)

    def data_point_to_bytes(self, data: t.Any) -> bytes:
        """Tranform a data point to its `bytes` representation"""
        if isinstance(data, PIL.Image.Image):
            return self._PIL_image_to_bytes_v0(data)
        else:
            raise Exception(f"Unsupported data point type: {type(data)}")

    def _PIL_image_to_bytes_v0(self, data: PIL.Image.Image) -> bytes:
        # Return raw image data
        return data.tobytes(encoder_name="raw")


class ARFFDataset(Dataset):
    """Dataset contained in an ARFF file"""

    def __init__(self, arff_file: t.Union[Path, str]):
        super().__init__(DatasetType.ARFF)
        self.arff_file = arff_file

    @property
    def data_points(self) -> t.Generator[bytes, None, None]:
        """Iterate through all data points (transformed as bytes)"""
        # The parsing is based on https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
        with open(self.arff_file, "rb") as f:
            reached_data = False
            for line in f:
                if reached_data:
                    yield line
                reached_data |= line.startswith((b"@data", b"@DATA"))


class CSVDataset(Dataset):
    """Dataset contained in a CSV file"""

    def __init__(
        self,
        csv_data: t.Union[t.Iterable[str], t.List[str], Path, str],
        dialect="excel",
        **fmtparams,
    ):
        super().__init__(DatasetType.CSV)
        if isinstance(
            csv_data,
            (
                Path,
                str,
            ),
        ):
            # Read from static file
            self.csv_data = open(csv_data, "r", newline="\n")
        else:
            # Already a list or iterable of rows
            self.csv_data = csv_data
        self.csv_reader = csv.reader(self.csv_data, dialect=dialect, **fmtparams)

    def __del__(self):
        if isinstance(self.csv_data, io.IOBase):
            # If data came from a file we opened, close the file
            self.csv_data.close()

    @property
    def data_points(self) -> t.Generator[bytes, None, None]:
        """Iterate through all data points (transformed as bytes)"""
        # FIXME(boyan): we must ignore the row that contains
        # the labels. The easiest way is to ignore the first row, but we
        # must be sure it's the one containing labels, if there are labels...
        for row in self.csv_reader:
            yield self._row_to_bytes_v0(row)

    def _row_to_bytes_v0(self, row: t.List[str]) -> bytes:
        """Tranform a data point to its `bytes` representation"""
        res = io.StringIO()
        writer = csv.writer(res, dialect=self.csv_reader.dialect)
        writer.writerow(row)
        return bytes(res.getvalue(), "utf-8")


class CanonicalDataset:
    """A format agnostic canonical dataset representation to compute fingerprints"""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        self.data_point_hashes: t.List[bytes] = []
        self._uid: DatasetUID | None = None
        self._fingerprint: dict[type, DatasetFingerprint | None] = {
            KeyedSha: None,
            Xor: None,
            SingleSha: None,
            Datasketch: None,
        }
        self.preprocessed: bool = False

        for d in dataset.data_points:
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

    def gen_fingerprint(self, algo: AlgoV0 = KeyedSha()) -> DatasetFingerprint:
        """Generate dataset fingerprint using given algorithm"""
        if self._fingerprint[type(algo)] is None:
            match algo:
                case KeyedSha(nb_signatures, magic_numbers) | Xor(
                    nb_signatures, magic_numbers
                ):
                    self._check_preprocess()
                    res = [None] * nb_signatures
                    for i in range(nb_signatures):
                        for h in self.data_point_hashes:
                            match algo:
                                case KeyedSha(_):
                                    h2 = hashlib.sha256(h + magic_numbers[i]).digest()
                                case Xor(_):
                                    h2 = [a ^ b for a, b in zip(h, magic_numbers[i])]
                            if res[i] is None or res[i] > h2:
                                res[i] = h2
                    self._fingerprint[type(algo)] = DatasetFingerprint(
                        res, MinHashType.MULTI
                    )
                case SingleSha(nb_signatures):
                    self._check_preprocess()
                    self._fingerprint[SingleSha] = DatasetFingerprint(
                        self.data_point_hashes[:nb_signatures], MinHashType.SINGLE
                    )
                case Datasketch():
                    res = MinHash()
                    for d in self.dataset.data_points:
                        res.update(d)
                    self._fingerprint[Datasketch] = DatasetFingerprint(
                        res, MinHashType.DATASKETCH
                    )
        return self._fingerprint[type(algo)]

    @property
    def keyed_sha_fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint based on keyed-SHA permutations.

        The figerprint is computed using the MinHash scheme.
        Random permutations are approximated by keyed SHA256 functions i.e. random bytes are appended to the input of SHA256.

        See https://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html, Min Hash with k hash functions.
        """
        return self.gen_fingerprint(algo=KeyedSha())

    @property
    def xor_fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint based on XOR permutations.

        The figerprint is computed using the MinHash scheme.
        Random permutations are approximated by the output of SHA256 XORed with random 256 bits values.
        """
        return self.gen_fingerprint(algo=Xor())

    @property
    def single_sha_fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint based on a single SHA permutation

        The figerprint is computed using the MinHash scheme with a single permutation approximated by SHA256.

        See https://web.eecs.utk.edu/~jplank/plank/classes/cs494/494/notes/Min-Hash/index.html, Min Hash with one hash functions.
        And https://en.wikipedia.org/wiki/MinHash#Variant_with_a_single_hash_function.
        """
        return self.gen_fingerprint(algo=SingleSha())

    @property
    def datasketch_fingerprint(self) -> DatasetFingerprint:
        """Generate dataset fingerprint based on universal hashing permutations

        The figerprint is computed using the MinHash scheme.
        Random permutations are approximated by universal hashing.

        See https://ekzhu.com/datasketch/minhash.html.
        """
        return self.gen_fingerprint(algo=Datasketch())
