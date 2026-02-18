from typing import Any, Protocol, Mapping
from collections.abc import Iterable, Iterator
from PIL import Image
from pathlib import Path
import io
import csv
from abc import ABC, abstractmethod
from collections.abc import Sequence
import struct
import numpy as np
from numpy.typing import ArrayLike

tf = None
tfds = None
try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except ImportError:
    pass


class IterableDataset(ABC):
    """Generic iterable dataset interface"""

    @abstractmethod
    def __iter__(self) -> Iterator[bytes]:
        """Get an iterator over the dataset"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifying the dataset"""
        pass

    @abstractmethod
    def serialize_data_point(self, data: Any) -> bytes:
        """
        Serialize a data point into bytes for it to be transmitted over
        a network interface
        """
        pass

    @abstractmethod
    def deserialize_data_point(self, data: bytes) -> Any:
        """
        Instantiate a data point from bytes that were generated using
        the `serialize_data_point` method
        """
        pass


class SequenceDataset(IterableDataset, ABC):
    """Generic indexable dataset interface"""

    @abstractmethod
    def __getitem__(self, idx: int) -> bytes:
        """Return an encoded data point by index"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Size of the dataset"""
        pass

    def __iter__(self) -> Iterator[bytes]:
        for i in range(len(self)):
            yield self[i]


# TODO(boyan): this should be versioned depending on the torchvision version as they might change their
# underlying format
# TODO(boyan): see whether we can use this class for torch datasets in general, or just torchvision.
class TorchVisionDataset(SequenceDataset):
    """Dataset loaded using the torch python API"""

    def __init__(self, dataset: Sequence[tuple[Image.Image, int]] | None = None):
        self.dataset = dataset

    def __getitem__(self, idx: int) -> bytes:
        if self.dataset is None:
            raise IndexError
        return self.serialize_data_point(self.dataset[idx])

    def __len__(self) -> int:
        return 0 if self.dataset is None else len(self.dataset)

    def __iter__(self) -> Iterator[bytes]:
        return (
            iter([])
            if self.dataset is None
            else (self.serialize_data_point(data_point) for data_point in iter(self.dataset))
        )

    @property
    def name(self) -> str:
        """Name identifying the dataset"""
        # In torch the dataset class names are descriptive like `MNIST`
        return type(self.dataset).__name__

    def serialize_data_point(self, data: tuple[Image.Image, int]) -> bytes:
        img, label = data
        width, height = img.size
        mode_bytes = img.mode.encode()
        image_bytes = img.tobytes(encoder_name="raw")

        return struct.pack(
            f"IIIII{len(mode_bytes)}s{len(image_bytes)}s",
            label,
            width,
            height,
            len(mode_bytes),
            len(image_bytes),
            mode_bytes,
            image_bytes,
        )

    def deserialize_data_point(self, data: bytes) -> tuple[Image.Image, int]:
        label, width, height, mode_len, image_len = struct.unpack_from("IIIII", data, offset=0)
        mode_bytes: bytes = struct.unpack_from(f"{mode_len}s", data, offset=20)[0]
        image_bytes: bytes = struct.unpack_from(f"{image_len}s", data, offset=20 + mode_len)[0]

        mode = mode_bytes.decode()
        size: tuple[int, int] = (width, height)

        return Image.frombytes(mode, size, image_bytes, decoder_name="raw"), label


class TfdsInfoProtocol(Protocol):
    features: Mapping[str, Any]


class TfdsDataset(IterableDataset):
    """Dataset loaded using tfds"""

    # There's a type checking issue without this trick
    _tf = tf
    _tfds = tfds

    def __init__(
        self, info: TfdsInfoProtocol, dataset: Iterable[dict[str, ArrayLike]] | None = None
    ):
        if TfdsDataset._tfds is None:
            raise RuntimeError(
                "TensorFlow Datasets not available. " "Install tfds deps: datasig[tfds]"
            )

        self.dataset = dataset
        self.features = list(info.features.keys())
        self.shapes = []
        self.dtypes = []
        self.byte_sizes = []

        for name in self.features:
            feat = info.features[name]
            if isinstance(
                feat,
                (
                    TfdsDataset._tfds.features.Tensor,
                    TfdsDataset._tfds.features.Image,
                    TfdsDataset._tfds.features.Audio,
                    TfdsDataset._tfds.features.Video,
                    TfdsDataset._tfds.features.Scalar,
                    TfdsDataset._tfds.features.ClassLabel,
                    TfdsDataset._tfds.features.BBoxFeature,
                ),
            ):
                shape = tuple(feat.shape)  # pyright: ignore
                dtype = np.dtype(feat.dtype.as_numpy_dtype)  # pyright: ignore

                self.dtypes.append(dtype)
                self.shapes.append(shape)
                self.byte_sizes.append(
                    int(np.prod([s for s in shape if s is not None]) * dtype.itemsize)
                    if None not in shape
                    else None  # pyright: ignore
                )
            else:
                raise RuntimeError(f"Unsupported TFDS feature type {type(feat)}")

    def __iter__(self) -> Iterator[bytes]:
        return (
            iter([])
            if self.dataset is None
            else (self.serialize_data_point(data_point) for data_point in iter(self.dataset))
        )

    @property
    def name(self) -> str:
        """Name identifying the dataset"""
        # In torch the dataset class names are descriptive like `MNIST`
        return type(self.dataset).__name__

    def serialize_data_point(self, data: dict[str, ArrayLike]) -> bytes:
        assert TfdsDataset._tf is not None
        format_str = ""
        args = []

        for name, size in zip(self.features, self.byte_sizes):
            feat = data[name]
            assert isinstance(feat, TfdsDataset._tf.Tensor)
            feat_bytes = np.asarray(feat).tobytes()
            if size is None:
                format_str += f"I{len(feat_bytes)}s"
                args += [len(feat_bytes), feat_bytes]
            else:
                assert size == len(feat_bytes)
                format_str += f"{size}s"
                args += [feat_bytes]

        return struct.pack(format_str, *args)

    def deserialize_data_point(self, data: bytes) -> dict[str, Any]:
        assert TfdsDataset._tf is not None
        offset = 0
        dict_data: dict[str, Any] = {}
        for name, size, shape, dtype in zip(
            self.features, self.byte_sizes, self.shapes, self.dtypes
        ):
            if size is None:
                feat_bytes_len: int = struct.unpack_from("I", data, offset=offset)[0]
                offset += 4
                feat_bytes: bytes = struct.unpack_from(f"{feat_bytes_len}s", data, offset=offset)[0]
                offset += feat_bytes_len
            else:
                feat_bytes: bytes = struct.unpack_from(f"{size}s", data, offset=offset)[0]
                offset += size
            dict_data[name] = TfdsDataset._tf.constant(
                np.frombuffer(feat_bytes, dtype=dtype).reshape(
                    tuple(-1 if s is None else s for s in shape)
                )
            )

        return dict_data


class ARFFDataset(IterableDataset):
    """Dataset contained in an ARFF file"""

    def __init__(self, arff_file: Path | str | None = None):
        self.arff_file = arff_file

    def __len__(self) -> int:
        """Return the number of data points in the dataset"""
        return sum(1 for _ in self.__iter__())

    def __iter__(self) -> Iterator[bytes]:
        """Iterate through all data points (transformed as bytes)"""
        if self.arff_file is None:
            return iter([])

        # The parsing is based on https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
        with open(self.arff_file, "rb") as f:
            reached_data = False
            for line in f:
                if reached_data:
                    yield line
                reached_data |= line.startswith((b"@data", b"@DATA"))

    @property
    def name(self) -> str:
        """Name identifying the dataset"""
        return str(self.arff_file)

    def serialize_data_point(self, data: bytes) -> bytes:
        # Data is already bytes
        return data

    def deserialize_data_point(self, data: bytes) -> bytes:
        return data


class CSVDataset(SequenceDataset):
    """Dataset contained in a CSV file"""

    def __init__(
        self,
        csv_data: Iterable[str] | list[str] | Path | str | None = None,
        dialect: str = "excel",
        **fmtparams: Any,
    ):
        self.csv_data = csv_data
        self.dialect = dialect
        self.fmtparams = fmtparams

        if isinstance(self.csv_data, (Path, str)):
            self._name = str(self.csv_data)
        else:
            # TODO(boyan): proper naming here needs to come from the user
            self._name = "Raw CSV data supplied from python"

        self.csv_reader = self._get_reader()
        self.cached_rows: list[list[str]] = []

    def _get_reader(self) -> Iterator[list[str]]:
        if self.csv_data is None:
            return iter([])

        if isinstance(self.csv_data, (Path, str)):
            # Read from static file
            _csv_data = open(self.csv_data, "r", newline="\n")
        else:
            # Already a list or iterable of rows
            _csv_data = self.csv_data
        return csv.reader(_csv_data, dialect=self.dialect, **self.fmtparams)

    def __del__(self):
        if isinstance(self.csv_data, io.IOBase):
            # If data came from a file we opened, close the file
            self.csv_data.close()

    @property
    def name(self) -> str:
        """Name identifying the dataset"""
        return self._name

    def serialize_data_point(self, data: list[str]) -> bytes:
        res = io.StringIO()
        writer = csv.writer(res)
        writer.writerow(data)
        return res.getvalue().encode()

    def deserialize_data_point(self, data: bytes) -> list[str]:
        reader = csv.reader(io.StringIO(data.decode()))
        return next(reader)

    def __len__(self) -> int:
        """Return the number of data points in the dataset"""
        return sum(1 for _ in self.__iter__())

    def __getitem__(self, idx: int) -> bytes:
        """
        This allows to access individual rows of the csv dataset.
        Note that it will result in the entire dataset being stored in
        memory.
        """
        # Iterate the csv reader until we get to the index
        remaining_steps = idx + 1 - len(self.cached_rows)
        for _ in range(0, remaining_steps):
            self.cached_rows.append(next(self.csv_reader))

        return self.serialize_data_point(self.cached_rows[idx])

    def __iter__(self) -> Iterator[bytes]:
        """Iterate through all data points (transformed as bytes)"""
        # FIXME(boyan): we must ignore the row that contains
        # the labels. The easiest way is to ignore the first row, but we
        # must be sure it's the one containing labels, if there are labels...
        for row in self._get_reader():
            yield self.serialize_data_point(row)
