import itertools as it
from typing import Any, Iterable, Tuple


def chunks(iterable: Iterable, batch_size: int) -> Iterable:
    """
    Divide an iterable into chunks of a specified size.

    Args:
        iterable (iterable): The iterable to divide into chunks.
        batch_size (int): The size of each chunk.

    Yields:
        tuple: The next chunk of the iterable as a tuple.

    """
    ite = iter(iterable)
    chunk = tuple(it.islice(ite, batch_size))

    while chunk:
        yield chunk
        chunk = tuple(it.islice(ite, batch_size))
