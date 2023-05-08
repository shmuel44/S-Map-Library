import math as mth
from typing import Optional, Tuple


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

    def lt(self) -> Tuple[int, int]:
        """
        Returns the point at the (left, top) location of the rectangle
        """
        return self.x, self.y

    def rb(self) -> Tuple[int, int]:
        """
        Returns the point at the (right, bottom) location of the rectangle
        """
        return self.x + self.width, self.y + self.height

    def ltrb(self) -> Tuple[int, int, int, int]:
        """
        Returns a tuple (left, top, right, bottom) representing the rectangle
        """
        return (*self.lt(), *self.rb())

    def size(self) -> Optional[Tuple[int, int]]:
        """
        Returns a tuple (width, height) representing the rectangle
        """
        return self.width, self.height
    
    def area(self) -> int:
        """Calculate the area of the rectangle.

        Returns:
            float: The area of the rectangle.
        """
        return self.width * self.height

    def contains(self, rectangle) -> bool:
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

    def grow(self, ratio, image_width, image_height) -> 'Rectangle':
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
            self.x - w,
            self.y - h,
            self.width + 2 * w,
            self.height + 2 * h)

        assert image.contains(new_rec)

        return new_rec
