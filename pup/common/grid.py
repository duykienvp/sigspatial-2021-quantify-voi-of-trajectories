"""
Grid related classes and functions
"""
import logging
import math
from collections import defaultdict
from typing import Optional

from pup.common import rectangle
from pup.common.coordinate_util import to_local_coordinates
from pup.common.datatypes import CheckinList
from pup.common.enums import AreaCode
from pup.common.rectangle import Rectangle


logger = logging.getLogger(__name__)


class Grid(Rectangle):
    """ An grid on the space. Grid cells are indexed from the min value. This grid covers the entire space

    Attributes
    ----------
    min_x: float
        minimum x value
    min_y: float
        minimum y value
    max_x: float
        maximum x value
    max_y: float
        maximum y value
    cell_len_x: float
        length of a cell in x dimension
    cell_len_y: float
        length of a cell in y dimension
    data: defaultdict
        value
    """

    min_x = None
    min_y = None
    max_x = None
    max_y = None

    cell_len_x = None
    cell_len_y = None

    def __init__(self, min_x, min_y, max_x, max_y, cell_len_x, cell_len_y):
        """
        Initialize a grid with given parameters.

        Parameters
        ----------
        min_x: float
            minimum x value
        min_y: float
            minimum y value
        max_x: float
            maximum x value
        max_y: float
            maximum y value
        cell_len_x: float
            length of a cell in x dimension
        cell_len_y: float
            length of a cell in y dimension
        """
        super().__init__(min_x, min_y, max_x, max_y)

        self.cell_len_x = float(cell_len_x)
        self.cell_len_y = float(cell_len_y)
        self.data = defaultdict(defaultdict)

    def get_shape(self) -> tuple:
        """
        Tuple of grid dimensions

        Returns
        -------
        tuple
            (shape_x, shape_y) as length in x and y dimensions
        """
        shape_x, shape_y = self.find_grid_index(self.max_x, self.max_y)

        # there are 2 cases:
        # case 1: when x size is divisible by the cell length, then nothing remains, so nothing needs to do
        # case 2: when x size is NOT divisible by the cell length, then there is still 1 remains, need to add 1 more
        if shape_x * self.cell_len_x < (self.max_x - self.min_x):
            shape_x += 1

        if shape_y * self.cell_len_y < (self.max_y - self.min_y):
            shape_y += 1

        return shape_x, shape_y

    def find_grid_index(self, x: float, y: float) -> tuple:
        """
        Find index in the grid of a position (x, y)

        Parameters
        ----------
        x: float
            x position
        y: float
            y position

        Returns
        -------
        tuple
            indexes in x and y dimensions
        """
        x_idx = find_grid_index_on_dimension(x, self.min_x, self.cell_len_x)
        y_idx = find_grid_index_on_dimension(y, self.min_y, self.cell_len_y)

        return x_idx, y_idx

    def find_cell_boundary(self, x_idx: int, y_idx: int) -> Optional[Rectangle]:
        """ Find the boundary of the cell (x_idx, y_idx)

        Parameters
        ----------
        x_idx
            index of cell in x dimension
        y_idx
            index of cell in y dimension

        Returns
        -------
        Rectangle
            boundary of the cell or None if the cell index is out of the grid
        """
        max_x_idx, max_y_idx = self.get_shape()
        if 0 <= x_idx < max_x_idx and 0 <= y_idx < max_y_idx:
            cell_min_x = x_idx * self.cell_len_x + self.min_x
            cell_min_y = y_idx * self.cell_len_y + self.min_y

            return Rectangle(cell_min_x, cell_min_y, cell_min_x + self.cell_len_x, cell_min_y + self.cell_len_y)

        else:
            logger.error('Cell index out of bound: ({}, {}) out of ({}, {})'.format(x_idx, y_idx, max_x_idx, max_y_idx))
            return None

    def extend_boundary_to_nearest(self, k: int):
        """
        Extend the boundary to the nearest k

        Parameters
        ----------
        k
            The order to extend to
        """
        self.min_x = extend_to_nearest(self.min_x, k, True)
        self.min_y = extend_to_nearest(self.min_y, k, True)
        self.max_x = extend_to_nearest(self.max_x, k, False)
        self.max_y = extend_to_nearest(self.max_y, k, False)

    def __str__(self):
        return "Grid(min_x={min_x}, min_y={min_y}, max_x={max_x}, " \
               "max_y={max_y}, cell_len_x={cell_len_x}, cell_len_y={cell_len_y})".format(**vars(self))


def find_grid_index_on_dimension(pos: float, min_value: float, cell_len: float) -> int:
    """

    Parameters
    ----------
    pos: float
        position
    min_value: float
        min value of the dimension
    cell_len: float
        size of each grid cell of the dimension
    Returns
    -------
    int
        grid index for pos
    """
    return int((pos - min_value) / cell_len)


def create_grid_for_data(data: CheckinList, cell_len_x: float, cell_len_y: float, boundary_order: int = 0) -> Grid:
    """
    Create a grid that covers the entire area.
    The boundary of grid can be extended to the nearest order

    Parameters
    ----------
    data
        list of check-ins
    cell_len_x
        length of a cell in x dimension
    cell_len_y
        length of a cell in y dimension
    boundary_order: optional
        Extend the boundary to the nearest boundary_order, 0 if not extend

    Returns
    -------
    Grid
        the grid that covers the entire area of the check-ins.
    """
    x = [c.x for c in data]
    y = [c.y for c in data]

    return create_grid(min(x), min(y), max(x), max(y), cell_len_x, cell_len_y, boundary_order)


def create_grid(min_x, min_y, max_x, max_y, cell_len_x, cell_len_y, boundary_order: int = 0) -> Grid:
    """
    Create a grid that covers the entire area of the check-ins.
    The boundary of grid can be extended to the nearest order

    Parameters
    ----------
    min_x: float
            minimum x value
    min_y: float
        minimum y value
    max_x: float
        maximum x value
    max_y: float
        maximum y value
    cell_len_x
        length of a cell in x dimension
    cell_len_y
        length of a cell in y dimension
    boundary_order: optional
        Extend the boundary to the nearest boundary_order, 0 if not extend

    Returns
    -------
    Grid
        the grid that covers the entire area of the check-ins.
    """
    grid = Grid(min_x, min_y, max_x, max_y, cell_len_x, cell_len_y)
    if boundary_order != 0:
        grid.extend_boundary_to_nearest(boundary_order)

    return grid


def extend_to_nearest(x: float, k: int, lower_bound: bool) -> float:
    """
    Extend a value x to the nearest lower/upper bound k.

    Parameters
    ----------
    x: float
        value to extend
    k: int
        order to extend to
    lower_bound: bool
        find lower bound or upper bound

    Returns
    -------
    float
        extended value of x

    Examples
    --------
    >>> extend_to_nearest(-24000, 5000, True)  # = -25000
    ... extend_to_nearest(24000, 5000, True)   # = 20000
    ... extend_to_nearest(0, 5000, True)       # = 0
    ... extend_to_nearest(-24000, 5000, False) # = -20000
    ... extend_to_nearest(24000, 5000, False)  # = 25000
    """
    v_floor = float(int(math.floor(abs(x) / float(k))) * k)
    v_ceil = float(int(math.ceil(abs(x) / float(k))) * k)
    if 0 <= x:
        if lower_bound:
            return v_floor
        else:
            return v_ceil
    else:
        if lower_bound:
            return -v_ceil
        else:
            return -v_floor


def create_grid_for_area(area_code: AreaCode, cell_len_x, cell_len_y, boundary_order: int = 0) -> Grid:
    """ Create a grid for an area

    Parameters
    ----------
    area_code
        area code
    cell_len_x
        length of a cell in x dimension
    cell_len_y
        length of a cell in y dimension
    boundary_order: optional
        Extend the boundary to the nearest boundary_order, 0 if not extend

    Returns
    -------
    Grid
        the grid for the area
    """
    min_x, min_y, max_x, max_y = get_boundary_for_area(area_code)
    return create_grid(min_x, min_y, max_x, max_y, cell_len_x, cell_len_y, boundary_order)


def get_boundary_for_area(area_code: AreaCode) -> tuple:
    """ Create a grid for an area.
    If an incorrect area is provided

    Parameters
    ----------
    area_code
        area code

    Returns
    -------
    tuple
        (min_x, min_y, max_x, max_y) boundary of this area
    """
    rect = rectangle.get_rectangle_for_area(area_code)
    orig_lon, orig_lat = rect.get_mid_point()  # latitude = y, longitude = x
    max_y, max_x = to_local_coordinates(orig_lat, orig_lon, rect.max_y, rect.max_x)
    min_y, min_x = to_local_coordinates(orig_lat, orig_lon, rect.min_y, rect.min_x)
    return min_x, min_y, max_x, max_y
