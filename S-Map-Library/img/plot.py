import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as pim



def display_image(pil_image: pim.Image, factor:int = 3):
    """
    Displays an image from a PIL image. The image is displayed with a
    figure size of 3 times the length and 2 times the width of the
    image.

    Args:
        pil_image (pim.Image): The PIL image to display.
        factor (int, optional): The factor to multiply the image size by. Defaults to 3.
    """
    fig = plt.figure(figsize=(factor*3, factor*2))

    plt.grid(False)
    plt.imshow(pil_image) # type: ignore
    axis = plt.gca()
    axis.get_xaxis().set_visible(False)  # disable x-axis
    axis.get_yaxis().set_visible(False)  # disable y-axis
    plt.show()



