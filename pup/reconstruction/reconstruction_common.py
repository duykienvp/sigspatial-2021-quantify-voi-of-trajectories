"""
Common method for reconstruction
"""
import logging
from typing import List

import numpy as np

from pup.common.datatypes import Traj
from pup.config import Config

logger = logging.getLogger(__name__)


def prepare_x_values(trajectory: Traj, indexes: List[int]) -> np.ndarray:
    """
    Prepare X values.

    :param trajectory: the input trajectory
    :param indexes: the list of indexes for select from
    :return: x values for prediction, of size n x 1
    """
    x_values = [trajectory[i].timestamp for i in indexes]
    x_values = np.asarray(x_values)
    x_values = x_values.reshape((-1, 1))

    return x_values


def prepare_y_values(trajectory: Traj, indexes: List[int]) -> np.ndarray:
    """
    Prepare y values which are (cy, cx) of data points at the selected indexes of the trajectory.

    :param trajectory: the input trajectory
    :param indexes: the list of indexes to select from
    :return:  y values for training as cy and cx values, of size n x 2
    """
    y_cy = list()
    y_cx = list()
    for i in indexes:
        y_cy.append(trajectory[i].y)
        y_cx.append(trajectory[i].x)

    y_cy = np.asarray(y_cy)
    y_cx = np.asarray(y_cx)
    y = np.column_stack((y_cy, y_cx))

    return y


def prepare_measurement_std(trajectory: Traj) -> float:
    """
    Prepare the measurement standard deviation which is the measurement standard deviation of the first measurement
    in the input trajectory or the default value from configuration if the trajectory is empty.

    :param trajectory: the input trajectory
    :return: the measurement standard deviation
    """
    if 0 < len(trajectory):
        return trajectory[0].measurement_std
    else:
        return Config.eval_default_location_measurement_std
