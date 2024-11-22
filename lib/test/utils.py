from ..dataset import CanonicalDataset

def assert_fingerprint_similarity(dataset_a: CanonicalDataset, dataset_b: CanonicalDataset, expected_similarity: float):
    fingerprint_a = dataset_a.fingerprint
    fingerprint_b = dataset_b.fingerprint
    assert len(fingerprint_a) == len(fingerprint_b)

    similarity = fingerprint_a.similarity(fingerprint_b)
    expected_error = 1.0-fingerprint_a.comparison_accuracy()
    assert similarity >= 0.0 and similarity <= 1.0
    assert similarity >= expected_similarity - expected_error
    assert similarity <= expected_similarity + expected_error