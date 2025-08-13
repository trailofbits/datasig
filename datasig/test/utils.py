from datasig.algo import KeyedShaMinHash
from datasig.dataset import IterableDataset
import requests
import arff # pyright: ignore[reportMissingTypeStubs]


def download_file(url: str, outfile: str | None = None) -> str:
    if not outfile:
        outfile = url.split("/")[-1]
    with requests.get(url) as r:
        with open(outfile, "wb") as f:
            f.write(r.content)
    return outfile


def assert_fingerprint_similarity(
    dataset_a: IterableDataset, dataset_b: IterableDataset, expected_similarity: float
):
    """Check dataset similarity using their fingerprints against an expected
    similarity value"""
    fingerprint_a = KeyedShaMinHash(dataset_a).digest()
    fingerprint_b = KeyedShaMinHash(dataset_b).digest()
    assert len(fingerprint_a) == len(fingerprint_b)

    similarity = fingerprint_a.similarity(fingerprint_b)
    expected_error = 1.0 - fingerprint_a.comparison_accuracy()
    print("SIM IS: ", similarity)
    assert similarity >= 0.0 and similarity <= 1.0
    assert similarity >= expected_similarity - expected_error
    assert similarity <= expected_similarity + expected_error


def extract_csv_range(csv_file: "str", start: int, end: int, outfile: str) -> str:
    """Extracts a range of rows of a CSV file into another CSV file.
    This assumes that each row is delimited by a newline, and that the first row is labels"""
    with open(csv_file, "rb") as f:
        with open(outfile, "wb") as outf:
            lines = f.readlines()
            outf.write(lines[0])
            for l in lines[start + 1 : end + 1]:  # +1 to ignore first row
                outf.write(l)
    return outfile


def extract_arff_range(arff_file: str, start: int, end: int, outfile: str) -> str:
    """Extracts a range of data points from an ARFF file into another ARFF file"""
    with open(arff_file, "r") as f:
        with open(outfile, "w") as outf:
            data = arff.load(f) # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            data["data"] = data["data"][start:end]
            outf.write(arff.dumps(data)) # pyright: ignore[reportUnknownMemberType]
    return outfile
