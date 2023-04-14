import itertools as it
from typing import Any, SupportsIter, Tuple


def chunks(iterable: SupportsIter, batch_size: int) -> Tuple[Any]:
    """
    Divide an iterable into chunks of a specified size.

    Args:
        iterable (iterable): The iterable to divide into chunks.
        batch_size (int): The size of each chunk.

    Yields:
        tuple: The next chunk of the iterable as a tuple.

    """
    it = iter(iterable)
    chunk = tuple(it.islice(it, batch_size))

    while chunk:
        yield chunk
        chunk = tuple(it.islice(it, batch_size))
