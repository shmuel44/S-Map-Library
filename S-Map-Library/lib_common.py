import pandas as pd
import itertools as it
from typing import List, Dict, Tuple, Iterable, Any, Union, Optional


def cleanup_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up dictionary values by converting pd.Timestamp objects to strings and replacing None with empty strings.
    
    Args:
        d (Dict[str, Any]): The input dictionary.

    Returns:
        Dict[str, Any]: The cleaned-up dictionary.
    """
    return {
        name: format_value(value)
        for name, value in d.items()
    }

def format_value(value: Any) -> Any:
    """
    Formats the given value for display.

    Args:
        value (Any): The value to format.

    Returns:
        Any: The formatted value.
    """
    if value is None:
        return '<<NULL>>'
    elif isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d %X')
    else:
        return value
    
def chunks(iterable: Iterable, batch_size: int) -> Iterable[Tuple]:
    """
    Divide an iterable into chunks of a specified size.

    Args:
        iterable (Iterable): The iterable to divide into chunks.
        batch_size (int): The size of each chunk.

    Yields:
        Tuple: The next chunk of the iterable as a tuple.
    """
    ite = iter(iterable)
    chunk = tuple(it.islice(ite, batch_size))

    while chunk:
        yield chunk
        chunk = tuple(it.islice(ite, batch_size))