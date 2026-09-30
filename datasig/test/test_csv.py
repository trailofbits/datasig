from pathlib import Path

from datasig.algo import KeyedShaMinHash, UID
from datasig.dataset import CSVDataset

from .utils import assert_fingerprint_similarity


def csv_rows(start: int, end: int) -> list[str]:
    return [
        f"{index},sensor-{index % 17},metric-{index % 5},{index / 10:.1f}\n"
        for index in range(start, end)
    ]


def write_csv(path: Path, start: int, end: int) -> Path:
    path.write_text("id,sensor,metric,value\n" + "".join(csv_rows(start, end)))
    return path


def test_csv_v0(tmp_path: Path):
    csv_file = write_csv(tmp_path / "data.csv", 0, 500)
    dataset = CSVDataset(csv_file, delimiter=",")

    fingerprint = KeyedShaMinHash(dataset).digest()
    uid = UID(CSVDataset(csv_file, delimiter=",")).digest()

    assert (
        uid
        == b"\x86\x02y\xa4(br\x18\x8e1\x0f\xcf\xfc`w1"
        b"\xde{\x01$HRK\xb5\xc5\xa10\x8c\n \xc1U"
    )
    assert len(fingerprint) == 400
    assert (
        fingerprint.signature()
        == KeyedShaMinHash(CSVDataset(csv_file, delimiter=",")).digest().signature()
    )


def test_similarity():
    d1 = CSVDataset(csv_rows(0, 5000))
    d2 = CSVDataset(csv_rows(5000, 10000))
    d3 = CSVDataset(csv_rows(0, 10000))
    d4 = CSVDataset(csv_rows(5000, 15000))

    # Identical datasets
    assert_fingerprint_similarity(d1, d1, 1.0)
    # Dataset vs. half dataset
    assert_fingerprint_similarity(d1, d3, 0.5)
    # Completely different datasets
    assert_fingerprint_similarity(d1, d2, 0.0)
    # 1/3rd in common
    assert_fingerprint_similarity(d3, d4, 0.33)


def test_serialization():
    data_point = [
        "166679",
        "375",
        "Nitrogen dioxide (NO2)",
        "Mean",
        "ppb",
        "CD",
        "414",
        "Rockaway and Broad Channel (CD14)",
        "Summer 2009",
        "06/01/2009",
        "8.44",
    ]
    serialized = CSVDataset().serialize_data_point(data_point)
    deserialized = CSVDataset().deserialize_data_point(serialized)
    assert data_point == deserialized
