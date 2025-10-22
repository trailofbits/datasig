from typing import Any
from collections.abc import Iterable, Iterator
from PIL import Image
from pathlib import Path
import io
import csv
from abc import ABC, abstractmethod
from collections.abc import Sequence
import struct


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

    def __init__(self, dataset: Sequence[tuple[Image.Image, int]]):
        self.dataset = dataset

    def __getitem__(self, idx: int) -> bytes:
        return TorchVisionDataset.serialize_data_point(self.dataset[idx])

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[bytes]:
        return (
            TorchVisionDataset.serialize_data_point(data_point) for data_point in iter(self.dataset)
        )

    @property
    def name(self) -> str:
        """Name identifying the dataset"""
        # In torch the dataset class names are descriptive like `MNIST`
        return type(self.dataset).__name__

    @staticmethod
    def serialize_data_point(data: tuple[Image.Image, int]) -> bytes:
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

    @staticmethod
    def deserialize_data_point(data: bytes) -> tuple[Image.Image, int]:
        label, width, height, mode_len, image_len = struct.unpack_from("IIIII", data, offset=0)
        mode_bytes: bytes = struct.unpack_from(f"{mode_len}s", data, offset=20)[0]
        image_bytes: bytes = struct.unpack_from(f"{image_len}s", data, offset=20 + mode_len)[0]

        mode = mode_bytes.decode()
        size: tuple[int, int] = (width, height)

        return Image.frombytes(mode, size, image_bytes, decoder_name="raw"), label


class ARFFDataset(IterableDataset):
    """Dataset contained in an ARFF file"""

    def __init__(self, arff_file: Path | str):
        self.arff_file = arff_file

    def __len__(self) -> int:
        """Return the number of data points in the dataset"""
        return sum(1 for _ in self.__iter__())

    def __iter__(self) -> Iterator[bytes]:
        """Iterate through all data points (transformed as bytes)"""
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

    @staticmethod
    def serialize_data_point(data: bytes) -> bytes:
        # Data is already bytes
        return data

    @staticmethod
    def deserialize_data_point(data: bytes) -> bytes:
        return data


class TensorFlowDataset(SequenceDataset):
    """Dataset loaded using the tensorflow_datasets python API"""

    def __init__(self, dataset: Sequence[dict[str, Any]]):
        """
        Initialize a TensorFlow dataset.

        Args:
            dataset: A tensorflow_datasets dataset that supports indexing and length.
                    Each sample should be a dictionary with keys like 'image' and 'label'.
        """
        self.dataset = dataset

    def __getitem__(self, idx: int) -> bytes:
        return TensorFlowDataset.serialize_data_point(self.dataset[idx])

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[bytes]:
        return (
            TensorFlowDataset.serialize_data_point(data_point) for data_point in iter(self.dataset)
        )

    @property
    def name(self) -> str:
        """Name identifying the dataset"""
        # TensorFlow datasets have a name attribute
        if hasattr(self.dataset, "name"):
            return self.dataset.name
        return "TensorFlowDataset"

    @staticmethod
    def serialize_data_point(data: dict[str, Any]) -> bytes:
        """
        Serialize a tensorflow dataset sample.

        Expected format: {'image': array, 'label': int}
        Serializes as: [label (4 bytes)][image_shape_len (4 bytes)][image_shape][image_dtype_len (4 bytes)][image_dtype][image_data]
        """
        # Handle both dict format and tuple format (image, label)
        if isinstance(data, dict):
            image = data["image"]
            label = int(data["label"])
        else:
            # Assume it's a tuple/list (image, label)
            image, label = data
            label = int(label)

        # Convert image to numpy array if needed
        import numpy as np
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Serialize the data
        image_bytes = image.tobytes()
        shape_bytes = str(image.shape).encode()
        dtype_bytes = str(image.dtype).encode()

        # Format: [label][shape_len][dtype_len][shape_bytes][image_len][dtype_bytes][image_bytes]
        # Use '=' to disable automatic padding alignment
        return struct.pack(
            f"=III{len(shape_bytes)}sI{len(dtype_bytes)}s{len(image_bytes)}s",
            label,
            len(shape_bytes),
            len(dtype_bytes),
            shape_bytes,
            len(image_bytes),
            dtype_bytes,
            image_bytes,
        )

    @staticmethod
    def deserialize_data_point(data: bytes) -> dict[str, Any]:
        """
        Deserialize a tensorflow dataset sample back to dict format.
        """
        import numpy as np

        # Unpack the header - use '=' to match serialization format
        label, shape_len, dtype_len = struct.unpack_from("=III", data, offset=0)
        offset = 12

        shape_bytes: bytes = struct.unpack_from(f"={shape_len}s", data, offset=offset)[0]
        offset += shape_len

        image_len: int = struct.unpack_from("=I", data, offset=offset)[0]
        offset += 4

        dtype_bytes: bytes = struct.unpack_from(f"={dtype_len}s", data, offset=offset)[0]
        offset += dtype_len

        image_bytes: bytes = struct.unpack_from(f"={image_len}s", data, offset=offset)[0]

        # Reconstruct the image
        shape = eval(shape_bytes.decode())  # Safe since we created this
        dtype = np.dtype(dtype_bytes.decode())
        image = np.frombuffer(image_bytes, dtype=dtype).reshape(shape)

        return {"image": image, "label": label}


class CSVDataset(SequenceDataset):
    """Dataset contained in a CSV file"""

    def __init__(
        self,
        csv_data: Iterable[str] | list[str] | Path | str,
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

    @staticmethod
    def serialize_data_point(data: list[str]) -> bytes:
        res = io.StringIO()
        writer = csv.writer(res)
        writer.writerow(data)
        return res.getvalue().encode()

    @staticmethod
    def deserialize_data_point(data: bytes) -> list[str]:
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

        return CSVDataset.serialize_data_point(self.cached_rows[idx])

    def __iter__(self) -> Iterator[bytes]:
        """Iterate through all data points (transformed as bytes)"""
        # FIXME(boyan): we must ignore the row that contains
        # the labels. The easiest way is to ignore the first row, but we
        # must be sure it's the one containing labels, if there are labels...
        for row in self._get_reader():
            yield CSVDataset.serialize_data_point(row)
