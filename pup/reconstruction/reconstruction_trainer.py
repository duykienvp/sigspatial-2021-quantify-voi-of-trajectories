"""
Train models (e.g., Gaussian Processes) for reconstruction
"""

import logging
from typing import List

import numpy as np
from tensorflow.python.framework.errors_impl import ResourceExhaustedError, InvalidArgumentError

from pup.common.datatypes import Traj
from pup.config import Config
from pup.reconstruction.gpr import gpr_location
from pup.reconstruction.reconstruction_common import prepare_x_values, prepare_measurement_std, prepare_y_values

logger = logging.getLogger(__name__)


def train_GP_model(user_id, traj_idx: int, trajectory: Traj, kernel_stds: List[float], kernel_variance_trainable: bool):
    """ We create separate GPs for latitude and longitude predictions.
    The X (or features) is the timestamp, the y (or label) is (cy, cx) values.
    The configuration for GP is retrieved from Config.

    :param user_id: user id of the trajectory
    :param traj_idx: trajectory index
    :param trajectory: trajectory
    """
    kernel_type = Config.reconstruction_gp_kernel
    framework_type = Config.reconstruction_gp_framework
    x_train = prepare_x_train(trajectory)
    y_train = prepare_y_train(trajectory)
    measurement_std = prepare_measurement_std(trajectory)

    try:

        models = gpr_location.fit_unsplit_unscaled(
            x_train, y_train, measurement_std, kernel_type, kernel_stds, kernel_variance_trainable, framework_type
        )

        return models

    except ResourceExhaustedError as err:
        logger.error("ResourceExhaustedError user {}, traj {}: ".format(user_id, traj_idx, err.message))
    except InvalidArgumentError as err:
        logger.error("InvalidArgumentError user {}, traj {}: {}, ".format(user_id, traj_idx, err.message))


def prepare_x_train(trajectory: Traj) -> np.ndarray:
    """
    Prepare X values for training, which are timestamps of data points.

    :param trajectory: the input trajectory
    :return: x values for training, of size n x 1
    """
    return prepare_x_values(trajectory, list(range(len(trajectory))))


def prepare_y_train(add_noise_trajectory: Traj) -> np.ndarray:
    """
    Prepare y values for training, which are (cy, cx) of data points in noise-added trajectory.

    :param add_noise_trajectory: the trajectory with noise added
    :return: x values for training, of size n x 1
    """
    return prepare_y_values(add_noise_trajectory, list(range(len(add_noise_trajectory))))
