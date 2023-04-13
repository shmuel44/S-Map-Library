import dataiku as di
import urllib.request as ul
from typing import List, Dict, Tuple, Iterable, Any, Union, Optional


def get_folder(folder: Union[str, di.Folder]) -> di.Folder:
    """Gets a managed folder handle.

    Args:
        folder (Union[str, di.Folder]): The name of the DSS managed folder, or the folder handle.

    Returns:
        dataiku.Folder: The managed folder handle.
    """
    return folder if isinstance(folder, di.Folder) else di.Folder(folder)

def path_exists(folder: Union[str, di.Folder], path: str) -> bool:
    """Checks whether a path exists in a managed folder.

    Args:
        folder (Union[str, di.Folder]): The name of the DSS managed folder, or the folder handle.
        path (str): The path to check existence for.

    Returns:
        bool: True if the path exists, False otherwise.
    """
    try:
        info = get_folder(folder).get_path_details(path)
        return info['exists'], info
    except:
        return False, None

def file_exists(folder: Union[str, di.Folder], path: str) -> bool:
    """Checks whether a path exists in a managed folder and is a file.

    Args:
        folder (Union[str, di.Folder]): The name of the DSS managed folder, or the folder handle.
        file_path (str): The path to the image file.

    Returns:
        bool: True if the path is a valid file path, False otherwise.
    """
    try:
        info = get_folder(folder).get_path_details(path)
        return not info['directory'] and info['exists'], info
    except:
        return False, None

def directory_exists(folder: Union[str, di.Folder], path: str) -> bool:
    """Checks whether a path exists in a managed folder and is a directory.

    Args:
        folder (Union[str, di.Folder]): The name of the DSS managed folder, or the folder handle.
        file_path (str): The path to the image file.

    Returns:
        bool: True if the path is a valid directory path, False otherwise.
    """
    try:
        info = get_folder(folder).get_path_details(path)
        return info['directory'] and info['exists'], info
    except:
        return False, None

def get_url(url: str, user_agent: str = None) -> bytes:
    """Downloads the contents of a URL.

    Args:
        url (str): The URL to download.
        user_agent (str, optional): The user agent string to use in the request. Defaults to None.

    Returns:
        bytes: The downloaded content as bytes.
    """
    opener = ul.build_opener()
    dummy_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'

    opener.addheaders = [
        (
            'User-Agent',
            dummy_agent if user_agent is None else user_agent
        )]

    ul.install_opener(opener)
    response = ul.urlopen(url)

    return response.read()

def url_to_managed_folder(url: str, folder: Union[str, di.Folder], path: str) -> bool:
    """Downloads the contents of a URL and saves it in a managed folder.

    Args:
        url (str): The URL to download.
        folder (Union[str, di.Folder]): The name of the DSS managed folder, or the folder handle.
        path (str): The path to the file.

    Returns:
        bool: True if the file exists after the copy, False otherwise.
    """
    exists = False
    
    try:
        handle = get_folder(folder)
        
        with handle.get_writer(path) as w:
            w.write(get_url(url))
            
        exists = file_exists(handle, path)
    except:
        exists = False
        
    return exists