import dataiku as di
import numpy as np
import PIL as pil
from PIL import Image as pim
import math as mth
from typing import List, Dict, Tuple, Iterable, Any, Union, Optional

import lib_io as lio
import lib_images as lim


class ImageContainer:
    """
    A class to represent an image container.

    Attributes:
        __folder (str): The folder containing the image.
        __file_path (str): The file path of the image.
        __width (int): The target width of the image.
        __height (int): The target height of the image.
    """

    def __init__(self, folder: Union[str, di.Folder], file_path: str, width: int, height: int):
        """
        Initializes an instance of the ImageContainer class.

        Args:
            folder (Union[str, di.Folder]): The folder containing the image.
            file_path (str): The file path of the image.
            width (int): The target width of the image.
            height (int): The target height of the image.
        """
        self.__handle = lio.get_folder(folder)
        self.__file_path = file_path
        self.__width = width
        self.__height = height
        self.__key = None
        self.__image = None
        self.__valid_image = False

    @staticmethod
    def from_directory(folder: Union[str, di.Folder], files: Optional[List[str]], width: int, height: int):
        handle = lio.get_folder(folder)
        file_list = handle.list_paths_in_partition() if files is None else files
        return [ImageContainer(handle, file, width, height) for file in file_list]

    def prepare(self):
        """
        Prepares the image by getting the key.
        """
        exists, info = lio.file_exists(self.__handle, self.__file_path)
        
        if exists:
            self.__key = info['name']
        else:
            self.__key = None
            
        return self.is_prepared()

    def validate(self):
        self.__valid_image = lim.is_valid_image(self.__handle, self.__file_path)
        return self.is_valid()
    
    @property
    def key(self):
        return self.__key

    @property
    def size(self):
        return (self.__width, self.__height)

    @property
    def image(self):
        if self.__image is None:
            if not self.is_prepared():
                raise Exception('The ImageContainer instance has not been prepared. Call prepared() first.')
                
            if not self.is_valid():
                raise Exception('The ImageContainer instance has not been validated. Call validate() first.')
                
            self.__image = lim.load_and_resize_image(
                folder=self.__handle,
                file_path=self.__file_path,
                width=self.__width,
                height=self.__height)
            
        return self.__image
            
    def is_prepared(self) -> bool:
        """
        Checks if the image is prepared.

        Returns:
            bool: True if the image is prepared, otherwise False.
        """
        return self.__key is not None
       
    def is_valid(self) -> bool:
        """
        Checks if the image is valid.

        Returns:
            bool: True if the image is valid, otherwise False.
        """
        return self.__valid_image

    def get_batch_array(self, rescaled: bool = True) -> Optional[np.ndarray]:
        """
        Converts the image to a batch array.

        Args:
            rescaled (bool, optional): Whether to rescale the image. Defaults to True.

        Returns:
            np.ndarray: The image as a batch array.
        """
        if self.image is None:
            return None
        return lim.image_to_batch_array(
            pil_image=self.image,
            rescaled=rescaled)


class Rectangle:
    """A rectangle with coordinates (x, y) and dimensions (width, height).

    Args:
        x (int): The x-coordinate of the top-left corner of the rectangle.
        y (int): The y-coordinate of the top-left corner of the rectangle.
        width (int): The width of the rectangle.
        height (int): The height of the rectangle.
    """

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f"Rectangle(x={self.x}, y={self.y}, width={self.width}, height={self.height})"

    def area(self) -> int:
        """Calculate the area of the rectangle.

        Returns:
            int: The area of the rectangle.
        """
        return self.width * self.height

    def contains(self, rectangle: 'Rectangle') -> bool:
        """Check if another rectangle is completely contained within this rectangle.

        Args:
            rectangle (Rectangle): The rectangle to check.

        Returns:
            bool: True if the given rectangle is contained within this rectangle, False otherwise.
        """
        return (self.x <= rectangle.x
                and self.y <= rectangle.y
                and self.x + self.width >= rectangle.x + rectangle.width
                and self.y + self.height >= rectangle.y + rectangle.height)

    def grow(self, ratio: float, image_width: int, image_height: int) -> 'Rectangle':
        """Try and inflate a rectangle by a ratio, without exceeding the image itself.

        Args:
            ratio (float): The ratio by which to inflate the rectangle.
            image_width (int): The width of the image containing the rectangle.
            image_height (int): The height of the image containing the rectangle.

        Returns:
            Rectangle: The grown rectangle, given the constraints.
        """
        image = Rectangle(0, 0, image_width, image_height)

        w_max = min(
            min(self.x, image.width - (self.x + self.width)),
            mth.floor(self.width * ratio / 2.0))

        h_max = min(
            min(self.y, image.height - (self.y + self.height)),
            mth.floor(self.height * ratio / 2.0))

        r_max = min(
            2 * w_max / self.width,
            2 * h_max / self.height)

        w = mth.floor(self.width * r_max / 2.0)

        h = mth.floor(self.height * r_max / 2.0)

        new_rec = Rectangle(
            int(self.x - w),
            int(self.y - h),
            int(self.width + 2 * w),
            int(self.height + 2 * h))

        assert image.contains(new_rec)

        return new_rec
        