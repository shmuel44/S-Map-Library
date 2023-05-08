import io as io
from typing import Any
from PIL import Image as pim
import urllib as ul



def get_url(url: str, user_agent: str = None) -> Any:
    """Downloads the contents of a URL.

    Args:
        url (str): The URL to download.
        user_agent (str, optional): The user agent string to use in the request. Defaults to None.

    Returns:
        bytes: The downloaded content as bytes.
    """
    opener = ul.request.build_opener()
    dummy_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'

    opener.addheaders = [
        (
            'User-Agent',
            dummy_agent if user_agent is None else user_agent
        )]

    ul.request.install_opener(opener)
    response = ul.request.urlopen(url)

    return response.read()


def download_bytes(url: str) -> io.BytesIO:
    """Downloads the contents of a URL as a BytesIO object.

    Args:
        url (str): The URL to download.

    Returns:
        io.BytesIO: The downloaded content as a BytesIO object.
    """
    return io.BytesIO(
        get_url(url))


def download_image(url: str) -> pim.Image:
    """Downloads an image from a URL and returns it as a PIL Image object.

    Args:
        url (str): The URL of the image to download.

    Returns:
        pim.Image: The downloaded image as a PIL Image object.
    """
    return pim.Image.open(
        download_bytes(
            url))


def download_text(url: str) -> str:
    """Downloads the contents of a URL as a string.

    Args:
        url (str): The URL to download.

    Returns:
        str: The downloaded content as a string.
    """
    return str(
        get_url(url))


def get_url_filename(url: str) -> str:
    """Extracts the filename from a URL.

    Args:
        url (str): The URL to extract the filename from.

    Returns:
        str: The filename extracted from the URL.
    """
    parts = url.split('/')
    return parts[-1]
