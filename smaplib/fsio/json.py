import json as js



def to_json(data):
    """Convert a Python object to a JSON string.

    Args:
        data (any): The Python object to convert.

    Returns:
        str: The JSON string.
    """
    return js.dumps(data, sort_keys=False, indent=4)


def save_to_json(file_path, data):
    """Save a Python object to a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        data (any): The Python object to save.

    Returns:
        str: The path to the saved file.
    """
    with open(file_path, 'w') as fp:
        js.dump(data, fp, sort_keys=False, indent=4)
    return file_path


def load_from_json(file_path):
    """Load a Python object from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        any: The loaded Python object.
    """
    with open(file_path, 'rb') as fp:
        data = js.load(fp)
    return data
