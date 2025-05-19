from time import process_time
from contextlib import contextmanager
from typing import Callable

# NOTE(boyan): taken from https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
@contextmanager
def catchtime() -> Callable[[], float]:
    # TODO(boyan): use thread_time?
    t1 = t2 = process_time()
    yield lambda: t2 - t1
    t2 = process_time()
