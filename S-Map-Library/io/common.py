import os as os
import hashlib as hl



def md5(file_path):
    """
    Calculates the MD5 hash of a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The MD5 hash of the file.
    """
    hash_md5 = hl.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_timestamped_filename(filename, extension, stamp):
    """
    Generates a filename with a timestamp.

    Args:
        filename (str): The filename without extension.
        extension (str): The file extension.
        stamp (datetime): The datetime to use as the timestamp.

    Returns:
        str: The timestamped filename.
    """
    return '{0}-{1}.{2}'.format(
        filename,
        get_stamp_text(stamp),
        extension)


def get_timestamped_name(name, stamp):
    """
    Generates a name with a timestamp.

    Args:
        name (str): The name without timestamp.
        stamp (datetime): The datetime to use as the timestamp.

    Returns:
        str: The timestamped name.
    """
    return '{0}-{1}'.format(
        name,
        get_stamp_text(stamp))


def get_stamp_text(stamp):
    """
    Generates a string representation of a timestamp.

    Args:
        stamp (datetime): The datetime to use as the timestamp.

    Returns:
        str: The timestamp as a string.
    """
    return '{0:04d}-{1:02d}-{2:02d}-{3:02d}-{4:02d}-{5:02d}'.format(
        stamp.year,
        stamp.month,
        stamp.day,
        stamp.hour,
        stamp.minute,
        stamp.second)


def create_if_not_exists(path):
    """Create a directory if it does not exist.

    Args:
        path (str): The path of the directory to create.

    Returns:
        str: The path of the created directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_stamp_folder(path, stamp):
    """Create a subdirectory with a timestamped name in the specified path.

    Args:
        path (str): The path of the parent directory.
        stamp (datetime.datetime): The timestamp to use for the subdirectory name.

    Returns:
        str: The path of the created subdirectory.
    """
    return create_if_not_exists(
        os.path.join(
            path,
            get_stamp_text(stamp)))


def get_path(path, file):
    """Get the full path to a file in a directory.

    Args:
        path (str): The path of the directory.
        file (str): The name of the file.

    Returns:
        str: The full path to the file.
    """
    file_path = os.path.join(path, file)
    file_exists = os.path.exists(file_path)
    print('[{0}] file "{1}" in path "{2}": "{3}"'.format('X' if file_exists else ' ', file, path, file_path))
    return file_path
