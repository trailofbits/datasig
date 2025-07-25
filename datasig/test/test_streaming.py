from torchvision.datasets import MNIST
from ..dataset import TorchVisionDataset, CanonicalDataset
from ..streaming import StreamedDataset
from ..config import ConfigV0
import pytest
import torch

# TODO(boyan): use properly generated tmp directories with tempfile
data = MNIST(root="/tmp/mnist_data", train=True, download=True)
help(data)

def test_dataset_full_iteration():
    dataset = TorchVisionDataset(data)
    stream = StreamedDataset(dataset)

    # Consume entire data stream
    count = 0
    for data_point in stream:
        count += 1

    # Check that we got the expected number of data points
    assert(len(dataset) == count)
    # Compare fingerprints, they should be the same
    c1 = CanonicalDataset(dataset, config=ConfigV0())
    c2 = CanonicalDataset(stream, config=ConfigV0())
    assert(c1.fingerprint == c2.fingerprint)

def test_dataset_partial_iteration():
    SUBSET_SIZE = 3000
    dataset = TorchVisionDataset(
        torch.utils.data.Subset(data, list(range(0, SUBSET_SIZE)))
    )
    stream = StreamedDataset(TorchVisionDataset(data))

    # Consume entire data stream
    count = 0
    for i, data_point in enumerate(stream):
        count += 1
        if i == SUBSET_SIZE-1:
            break


    # Check that we got the expected number of data points
    assert(len(dataset) == count)
    # Compare fingerprints, they should be the same
    c1 = CanonicalDataset(dataset, config=ConfigV0())
    c2 = CanonicalDataset(stream, config=ConfigV0())
    assert(c1.fingerprint == c2.fingerprint)


def test_dataset_indexes():
    subset_idxs = list(range(1, 4000, 3))
    dataset = TorchVisionDataset(
        torch.utils.data.Subset(data, subset_idxs)
    )
    stream = StreamedDataset(TorchVisionDataset(data))

    # Consume entire data stream
    for idx in subset_idxs:
        data_point = stream[idx]

    # Compare fingerprints, they should be the same
    c1 = CanonicalDataset(dataset, config=ConfigV0())
    c2 = CanonicalDataset(stream, config=ConfigV0())
    assert(c1.fingerprint == c2.fingerprint)