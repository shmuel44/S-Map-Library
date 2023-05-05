import pandas as pd
from typing import Dict, Any, List, Literal, Tuple
import numpy as np
import re
from urllib.parse import urlparse
from enum import Enum


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


def flatten_name_value_pairs(pairs: List[Tuple[str, Any, str]], separator: str) -> str:
    """Formats a list of name-value pairs into a flattened string with a separator between each pair.

    Args:
        pairs (List[Tuple[str, Any, str]]): A list of tuples, where each tuple contains the name, value, and format specifier for a pair to be formatted.
        separator (str): The separator to place between each name-value pair.

    Returns:
        str: A flattened string of name-value pairs with the specified separator between each pair.
    """
    message = ''

    for name, value, format in pairs:
        message += '{0}: {1}{2}'.format(
            name,
            ('{0:'+format+'}').format(value),
            separator)
            
    return message[:-1*len(separator)]


def shift_array(ar: np.ndarray, n:int) -> np.ndarray:
    """Shifts the elements of a 1D NumPy array by a specified number of positions.

    Args:
        ar (np.ndarray): The 1D NumPy array to be shifted.
        n (int): The number of positions by which to shift the array. A positive value will shift the array to the right, and a negative value will shift the array to the left.

    Returns:
        np.ndarray: The shifted 1D NumPy array, with empty positions filled with NaN values.
    """
    e = np.empty_like(ar)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = ar[:-n]
    else:
        e[n:] = np.nan
        e[:n] = ar[-n:]
    return e


class StringKind(Enum):
    """An enumeration of the kinds of strings (URL, file path, or other).
    """
    Url = 'Url'
    Path = 'Path'
    Other = 'Other'

def get_string_kind(string: str) -> StringKind:
    """Determines the kind of string (URL, file path, or other).
    Args:
        string (str): The string to be evaluated.
    Returns:
        StringKind: The kind of string.
    """
    try:
        url = urlparse(string)

        if url.scheme and url.netloc:
            return StringKind.Url
        
        file_path_pattern = re.compile(r'^(?:[a-zA-Z]:|\/|[a-zA-Z0-9_\-\s]+)(?:\/[a-zA-Z0-9_\-\s]+)*\.{0,1}[a-zA-Z0-9]*$')

        if file_path_pattern.match(string):
            return StringKind.Path
    except:
        pass
    
    return StringKind.Other

