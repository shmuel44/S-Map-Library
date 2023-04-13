import itertools as it



def chunks(iterable, batch_size):
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