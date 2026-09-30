"""Regression tests for the bottom-k dataset fingerprint."""

from hashlib import sha256

import pytest

from datasig.algo import SingleShaMinHash


def slots(signature: bytes) -> list[bytes]:
    return [signature[index : index + 32] for index in range(0, len(signature), 32)]


@pytest.mark.parametrize("width", [4, 400])
def test_repeated_low_hash_cannot_hide_an_added_row(width: int) -> None:
    candidates = [f"row:{index}".encode() for index in range(width + 1)]
    ordered = sorted(candidates, key=lambda row: sha256(row).digest())
    padding, added = ordered[:2]
    original = [padding, *ordered[2:], *([padding] * width)]
    changed = original + [added]
    original_signature = SingleShaMinHash(original, nb_signatures=width).digest().signature()
    changed_signature = SingleShaMinHash(changed, nb_signatures=width).digest().signature()

    assert original_signature != changed_signature
    assert len(set(slots(original_signature))) == width
    assert len(set(slots(changed_signature))) == width
    assert sha256(added).digest() in slots(changed_signature)


def test_repeated_rows_do_not_satisfy_distinct_requirement_and_updates_work():
    sketch = SingleShaMinHash([b"repeated"] * 4, nb_signatures=2)
    with pytest.raises(ValueError, match="Not enough distinct data points"):
        sketch.digest()

    sketch.update([b"repeated", b"new"])
    updated = sketch.digest().signature()
    assert slots(updated) == sorted({sha256(b"repeated").digest(), sha256(b"new").digest()})


def test_fewer_observations_than_requested_still_raise():
    with pytest.raises(ValueError, match="Not enough distinct data points"):
        SingleShaMinHash([b"only one"], nb_signatures=4).digest()


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="Not enough distinct data points"):
        SingleShaMinHash().digest()
