import itertools
from collections.abc import Iterable, Iterator, Sequence


def range_data[T](data: Iterable[T], start_offset: int, end_offset: int | None) -> Iterator[T]:
    return itertools.islice(data, start_offset, end_offset)


def offset_data[T](data: Iterable[T], offset: int) -> Iterator[T]:
    return range_data(data, offset, -1)


def half_data[T](data: Sequence[T]) -> Iterator[T]:
    return offset_data(data, len(data) // 2)


def concat_data[T](*iterables: Iterable[T]) -> Iterator[T]:
    return itertools.chain(*iterables)
