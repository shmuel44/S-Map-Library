import math as mth
import numpy as np
from PIL import Image as pim
from PIL import ImageOps as pio
import os as os
from typing import List, Tuple, Optional

from . import classes as cls



def fit_image(
    pil_image: pim.Image, 
    width: Optional[int], 
    height: Optional[int]) -> pim.Image:
    """Resizes an image to the given maximum dimension while preserving aspect ratio.

    Args:
        pil_image (pim.Image): The image to resize.
        width (Optional[int]): The desired width of the image.
        height (Optional[int]): The desired height of the image.

    Returns:
        pim.Image: The resized image.
    """
    if width is None and height is None:
        return pil_image
    
    if width is None:
        width = pil_image.width
        
    if height is None: 
        height = pil_image.height

    return pio.fit(
        pil_image,
        (width, height),
        pim.ANTIALIAS)


def resize_image(
    pil_image: pim.Image, 
    width: Optional[int], 
    height: Optional[int]) -> pim.Image:
    """Resizes an image to the given width and height.

    Args:
        pil_image (pim.Image): The image to resize.
        width (Optional[int]): The desired width of the image.
        height (Optional[int]): The desired height of the image.

    Returns:
        pim.Image: The resized image.
    """
    if width is None and height is None:
        return pil_image
    
    if width is None:
        width = pil_image.width
        
    if height is None: 
        height = pil_image.height
        
    return pil_image.resize(
        size=(width, height), 
        resample=pim.ANTIALIAS)


def image_to_array(
    pil_image: pim.Image, 
    rescaled: bool = True) -> np.ndarray:
    """Convert a PIL Image to a NumPy array.

    Args:
        pil_image (pim.Image): The input PIL Image.
        rescaled (bool, optional): Whether or not to rescale the array.

    Returns:
        np.ndarray: The NumPy array.
    """
    return np.array(pil_image, dtype=np.float32) / (255.0 if rescaled else 1.0)


def image_to_batch_array(
    pil_image: pim.Image, 
    rescaled: bool = True) -> np.ndarray:
    """Convert a PIL Image to a batched NumPy array.

    Args:
        pil_image (pim.Image): The input PIL Image.
        rescaled (bool, optional): Whether or not to rescale the array.

    Returns:
        np.ndarray: The batched NumPy array.
    """
    return image_to_array(pil_image=pil_image, rescaled=rescaled)[np.newaxis, :, :, :]


def images_to_batch_array(
    pil_images: List[pim.Image], 
    rescaled: bool = True) -> np.ndarray:
    """Convert a list of PIL Images to a batched NumPy array.

    Args:
        pil_images (List[pim.Image]): The list of PIL Images.
        rescaled (bool, optional): Whether or not to rescale the array.

    Returns:
        np.ndarray: The batched NumPy array.
    """
    ar = []

    for pil_image in pil_images:
        ar.append(image_to_array(pil_image, rescaled))

    return np.array(ar)


def crop_image(
    pil_image: pim.Image, 
    rectangle: cls.Rectangle) -> pim.Image:
    """Crop a PIL Image to a specified rectangle.

    Args:
        pil_image (pim.Image): The input PIL Image.
        rectangle (cim.Rectangle): The rectangle to crop.

    Returns:
        pim.Image: The cropped PIL Image.
    """
    return pil_image.transform(
        size=(rectangle.width, rectangle. height),
        method=pim.EXTENT,
        resample=pim.ANTIALIAS,
        data=rectangle.ltrb())


def extract_square_portion(
    image: pim.Image, 
    horizontal_position: Optional[str], 
    vertical_position: Optional[str], 
    output_size: Optional[Tuple[int, int]]) -> pim.Image:
    """
    Extracts a square portion of a PIL Image.

    Args:
        image (pim.Image): The input PIL Image.
        horizontal_position (Optional[str]): The horizontal position of the portion (left, center, right).
        vertical_position (Optional[str]): The vertical position of the portion (top, middle, bottom).
        output_size (Optional[Tuple[int, int]]): The desired output size.

    Returns:
        pim.Image: The extracted square portion.
    """
    max_dimension = min(image.width, image.height)

    # Determine the x coordinate of the left edge of the square portion based on the horizontal position.
    if horizontal_position == 'left':
        x = 0
    elif horizontal_position == 'center' or horizontal_position is None:
        x = mth.floor((image.width - max_dimension) / 2)
    elif horizontal_position == 'right':
        x = image.width - max_dimension
    else:
        raise ValueError(f'Invalid horizontal position: {horizontal_position}')

    # Determine the y coordinate of the top edge of the square portion based on the vertical position.
    if vertical_position == 'top':
        y = 0
    elif vertical_position == 'middle' or vertical_position is None:
        y = mth.floor((image.height - max_dimension) / 2)
    elif vertical_position == 'bottom':
        y = image.height - max_dimension
    else:
        raise ValueError(f'Invalid vertical position: {vertical_position}')

    # If the output size is not specified, use the maximum dimension of the square portion.
    if output_size is None:
        output_size = (max_dimension, max_dimension)

    # Crop the image to the square portion.
    return image.transform(
        size=output_size,
        method=pim.EXTENT,
        resample=pim.ANTIALIAS,
        data=(x, y, x + max_dimension, y + max_dimension))


def open_image(file_path: str) -> pim.Image:
    """Opens an image from the given file path.

    Args:
        file_path (str): The path to the image file.

    Returns:
        np.ndarray: The loaded image.
    """
    return pim.open(file_path)


def is_valid_image(file_path: str) -> bool:
    """
    Checks if the given file path corresponds to a valid image file.

    Args:
        file_path (str): The path to the file to check.

    Returns:
        bool: True if the image is valid, False otherwise.
    """
    try:
        # Open the image file.
        image = open_image(file_path)

        # Verify that the image file is a valid format and can be decoded.
        image.verify()

        # Check that the image dimensions are positive and non-zero.
        width, height = image.size
        if width <= 0 or height <= 0:
            return False

        return True

    except (IOError, SyntaxError, ValueError, OSError):
        # Catch specific exceptions that can occur when verifying the image file.
        return False


def load_image(file_path: str) -> np.ndarray:
    """Loads an image from the given file path.

    Args:
        file_path (str): The path to the image file.

    Returns:
        Image: The loaded image.
    """
    return np.array(
        open_image(
            file_path))


def load_and_fit_image(file_path: str, width: Optional[int], height: Optional[int]) -> pim.Image:
    """Loads an image and scales it to the given width and height while preserving aspect ratio.

    Args:
        file_path (str): The path to the image file.
        width (Optional[int]): The desired width of the image.
        height (Optional[int]): The desired height of the image.

    Returns:
        Image: The loaded and scaled image.
    """
    return fit_image(
        open_image(file_path),
        width, 
        height)


def load_and_resize_image(file_path: str, width: Optional[int], height: Optional[int]) -> pim.Image:
    """Loads an image and resizes it to the given width and height.

    Args:
        file_path (str): The path to the image file.
        width (Optional[int]): The desired width of the image.
        height (Optional[int]): The desired height of the image.

    Returns:
        pim.Image: The loaded and resized image.
    """
    return resize_image(
        open_image(file_path),
        width, 
        height)
