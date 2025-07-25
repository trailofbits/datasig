from .dataset import Dataset, RawDataset
from typing import Any, Generator


class StreamedDataset(Dataset):
    """
    This class is a wrapper around a Dataset that allows streaming the
    data. It can be used to stream the entire dataset, or only a subset of its
    data, and generate a fingerprint only for the data that was streamed, not the
    entire dataset. Under the hood, this class uses a RawDataset instance and for
    each data point streamed from the actual dataset, it adds the corresponding encoded
    data point to the raw dataset. Then the raw dataset is used to compute the fingerprint.

    - In order to stream the data, it implements the __getitem__ and __iter__ methods.
    - This class implements the `Dataset` class but with a few gotchas. Its
      `data_points` property are not available. This aims at enforcing the use of
      __getitem__ and __iter__ to stream the data.
    - This class overrides the `encoded_data_points` property so that it just returns
      the streamed data points that have been stored in the internal RawDataset instance.
    """

    def __init__(self, dataset: Dataset):
        super().__init__(dataset.type)
        self.dataset = dataset
        self.streamed_data_points: set[int] = set()
        self.streamed_data = RawDataset(self.dataset.name)

    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def data_points(self) -> Generator[any, None, None]:
        raise NotImplemented("This method should never be called on this class")

    def encode_data_point(self, data_point: Any) -> bytes:
        return self.dataset.encode_data_point(data_point)

    @property
    def encoded_data_points(self) -> Generator[bytes, None, None]:
        """This returns only the encodings for the data points that
        have effectively been streamed"""
        return self.streamed_data.encoded_data_points

    def record_streamed_data_point(self, data_point: Any):
        """
        Record a data point as streamed. We encode it to bytes
        using the encoding method of the underlying dataset being
        streamed, and then add it to the RawDataset in streamed_data
        """
        self.streamed_data.add_data_point(self.dataset.encode_data_point(data_point))

    def __getitem__(self, idx: int):
        """
        Stream an arbitrary data point by index. The returned data
        point is recorded and will be accounted for in fingerprint
        generation
        """
        # Get data point
        data_point = self.dataset[idx]
        # If first time streaming it, record it for later fingerprint
        if idx not in self.streamed_data_points:
            self.record_streamed_data_point(data_point)
            self.streamed_data_points.add(idx)
        return data_point

    def __iter__(self) -> Generator[Any, None, None]:
        """Return an iterator that goes over the data points of the underlying
        dataset being streamed. All data points retrieved from the generator are
        recorded and accounted for in fingerprint generation
        """
        return StreamedDatasetGenerator(self)

    def serialize_data_point(self, data: Any) -> bytes:
        return self.dataset.serialize_data_point(data)

    def deserialize_data_point(self, data: bytes) -> Any:
        return self.dataset.deserialize_data_point(data)


class StreamedDatasetGenerator:
    """A custom iterator for a streamed dataset"""

    def __init__(self, streamed_dataset):
        self.data_points_gen = streamed_dataset.dataset.data_points
        self.streamed_dataset = streamed_dataset

    def __next__(self):
        # This will raise the StopIteration exception when done
        d = next(self.data_points_gen)
        # NOTE(boyan): if the generator is created multiple times
        # we will add the same data point multiple times here.
        # That shouldn't be an issue other than performance but
        # should be handled if possible IMO.
        self.streamed_dataset.record_streamed_data_point(d)
        return d
