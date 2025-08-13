from datasig.dataset import ARFFDataset
from datasig.algo import KeyedShaMinHash, UID
from .utils import assert_fingerprint_similarity, download_file, extract_arff_range
from pprint import pprint
import pytest

# TODO(boyan): use properly generated tmp directories with tempfile
arff_file = download_file(
    "https://www.openml.org/data/download/2169/BayesianNetworkGenerator_trains.arff",
    "/tmp/data.arff",
)


@pytest.mark.skip(reason="Dataset unavailable")
def test_arff_v0():
    dataset = ARFFDataset(arff_file)
    fingerprint = KeyedShaMinHash(dataset).digest()
    uid = UID(dataset).digest()
    # Test UID and fingerprint
    pprint(uid, width=70)
    pprint(fingerprint.signature(), width=70)
    assert (
        uid
        == b""
    )
    assert len(fingerprint) == 400
    assert fingerprint.signature() == b""


@pytest.mark.skip(reason="Dataset unavailable")
def test_similarity():
    data1 = extract_arff_range(arff_file, 0, 5000, "/tmp/data1.arff")
    data2 = extract_arff_range(arff_file, 5000, 10000, "/tmp/data2.arff")
    data3 = extract_arff_range(arff_file, 0, 10000, "/tmp/data3.arff")
    data4 = extract_arff_range(arff_file, 5000, 15000, "/tmp/data4.arff")

    d1 = ARFFDataset(data1)
    d2 = ARFFDataset(data2)
    d3 = ARFFDataset(data3)
    d4 = ARFFDataset(data4)

    # Identical datasets
    assert_fingerprint_similarity(d1, d1, 1.0)
    # Dataset vs. half dataset
    assert_fingerprint_similarity(d1, d3, 0.5)
    # Completely different datasets
    assert_fingerprint_similarity(d1, d2, 0.0)
    # 1/3th in common
    assert_fingerprint_similarity(d3, d4, 0.33)


def test_serialization():
    data_point = b"A, 1234, Test data"
    serialized = ARFFDataset.serialize_data_point(data_point)
    deserialized = ARFFDataset.deserialize_data_point(serialized)

    assert data_point == deserialized
