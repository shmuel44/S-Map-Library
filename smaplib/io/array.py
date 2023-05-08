import numpy as np



def save_array(file_path, data):
    """Save a NumPy array to disk as a compressed .npz file.

    Args:
        file_path (str): The path to save the file to.
        data (numpy.ndarray): The NumPy array to save.

    Returns:
        str: The path of the saved file.
    """
    np.savez_compressed(file=file_path, x=data)
    return file_path


def load_array(file_path):
    """Load a NumPy array from a .npz file.

    Args:
        file_path (str): The path of the file to load.

    Returns:
        numpy.ndarray: The loaded NumPy array.
    """
    return np.load(file=file_path, allow_pickle=True)['x']


def save_cache_dict_file(file_path, dict):
    """Save a dictionary as a .npz file.

    Args:
        file_path (str): The path to save the file to.
        dict (dict): The dictionary to save.

    Returns:
        str: The path of the saved file.
    """
    return save_array(file_path=file_path, data=np.array(dict, dtype=object))


def load_cache_dict_file(file_path):
    """Load a dictionary from a .npz file.

    Args:
        file_path (str): The path of the file to load.

    Returns:
        dict: The loaded dictionary.
    """
    return load_array(file_path=file_path)[()]


def load_from_npz(file_path):
    """Load a .npz file and return its contents as a dictionary.

    Args:
        file_path (str): The path of the file to load.

    Returns:
        dict: The contents of the .npz file.
    """
    with np.load(file_path, allow_pickle=True) as npz_file:
        data = dict(npz_file.items())
    return data


def save_to_npz(file_path, data_dict):
    """Save a dictionary to disk as a compressed .npz file.

    Args:
        file_path (str): The path to save the file to.
        data_dict (dict): The dictionary to save.

    Returns:
        str: The path of the saved file.
    """
    np.savez_compressed(file_path, **data_dict)
    return file_path
