import itertools


def range_data(data, start_offset: int, end_offset: int):
    return iterools.islice(data, start_offset, end_offset)


def offset_data(data, offset: int):
    return range_data(data, offset, -1)


def half_data(data):
    return offset_data(data, len(data) / 2)


def concat_data(*iterables):
    return itertools.chain(iterables)
