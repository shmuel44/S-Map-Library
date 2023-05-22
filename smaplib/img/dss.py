import dataiku as di
import numpy as np
from typing import Iterable, List, Tuple, Union, Optional
from PIL import Image as pim

from . import helpers as lim
from ..io import dss as lio
from ..misc import iterable as it


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
        """_summary_

        Args:
            folder (Union[str, di.Folder]): _description_
            files (Optional[List[str]]): _description_
            width (int): _description_
            height (int): _description_

        Returns:
            _type_: _description_
        """
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
        """_summary_

        Returns:
            _type_: _description_
        """
        self.__valid_image = is_valid_image(self.__handle, self.__file_path)
        return self.is_valid()
    
    @property
    def key(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return self.__key

    @property
    def size(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return (self.__width, self.__height)

    @property
    def image(self):
        if self.__image is None:
            if not self.is_prepared():
                raise Exception('The ImageContainer instance has not been prepared. Call prepared() first.')
                
            if not self.is_valid():
                raise Exception('The ImageContainer instance has not been validated. Call validate() first.')
                
            self.__image = load_and_resize_image(
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


def open_valid_image(folder: Union[str, di.Folder], file_path: str) -> Optional[pim.Image]:
    """Opens a valid image from the given folder and file path.

    Args:
        folder (Union[str, di.Folder]): The name of the DSS managed folder, or the folder handle.
        file_path (str): The path to the image file.

    Returns:
        np.ndarray: The loaded image.
    """

    handle = lio.get_folder(folder)

    if handle is not None:
        if is_valid_image(handle, file_path):
            return open_image(handle, file_path)

    return None

def open_image(folder: Union[str, di.Folder], file_path: str) -> pim.Image:
    """Opens an image from the given folder and file path.

    Args:
        folder (Union[str, di.Folder]): The name of the DSS managed folder, or the folder handle.
        file_path (str): The path to the image file.

    Returns:
        np.ndarray: The loaded image.
    """
    with lio.get_folder(folder).get_download_stream(file_path) as stream:
        return pim.open(stream)

def is_valid_image(folder: Union[str, di.Folder], file_path: str) -> bool:
    """
    Checks if the given file path corresponds to a valid image file.

    Args:
        folder (Union[str, di.Folder]): The name of the DSS managed folder, or the folder handle.
        file_path (str): The path to the file to check.

    Returns:
        bool: True if the image is valid, False otherwise.
    """
    try:
        # Open the image file.
        image = open_image(folder, file_path)

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

def load_image(folder: Union[str, di.Folder], file_path: str) -> np.ndarray:
    """Loads an image from the given folder and file path.

    Args:
        folder (Union[str, di.Folder]): The name of the DSS managed folder, or the folder handle.
        file_path (str): The path to the image file.

    Returns:
        Image: The loaded image.
    """
    return np.array(
        open_image(
            folder,
            file_path))

def load_and_fit_image(folder: Union[str, di.Folder], file_path: str, width: Optional[int], height: Optional[int]) -> pim.Image:
    """Loads an image and scales it to the given width and height while preserving aspect ratio.

    Args:
        folder (Union[str, di.Folder]): The name of the DSS managed folder, or the folder handle.
        file_path (str): The path to the image file.
        width (Optional[int]): The desired width of the image.
        height (Optional[int]): The desired height of the image.

    Returns:
        Image: The loaded and scaled image.
    """
    return lim.fit_image(
        open_image(folder, file_path),
        width, 
        height)

def load_and_resize_image(folder: Union[str, di.Folder], file_path: str, width: Optional[int], height: Optional[int]) -> pim.Image:
    """Loads an image and resizes it to the given width and height.

    Args:
        folder (str or di.Folder): The name of the DSS managed folder, or the folder handle.
        file_path (str): The path to the image file.
        width (Optional[int]): The desired width of the image.
        height (Optional[int]): The desired height of the image.

    Returns:
        pim.Image: The loaded and resized image.
    """
    return lim.resize_image(
        open_image(folder, file_path),
        width, 
        height)
    
def get_images_as_batches(folder: Union[str, di.Folder], target_width: int, target_height: int, batch_size: int, rescaled: bool = True) -> Iterable[Tuple[np.ndarray, List[str]]]:
    """Returns image batches and file names for all images in a given directory.

    Args:
        images_path (str): The path to the directory containing the images.
        target_width (int): The desired width of the images.
        target_height (int): The desired height of the images.
        batch_size (int): The size of the batches to return.
        rescaled (bool, optional): Whether to rescale the images to the range [0,1]. Defaults to True.

    Yields:
        Tuple[numpy.ndarray, List[str]]: A tuple containing a batch of images and a list of their file names.
    """
    handle = lio.get_folder(folder)
    
    files = [file_path for file_path in handle.list_paths_in_partition()]
    
    for files_chunk in it.chunks(files, batch_size):
        images = []
        file_names = []
        
        for file in files_chunk:
            
            if is_valid_image(handle, file):
                image = open_image(handle, file)
            
                images.append(
                    lim.resize_image(image, target_width, target_height))
                
                file_names.append(
                    handle.get_path_details(file)['name'])

        yield lim.images_to_batch_array(pil_images=images, rescaled=rescaled), file_names


