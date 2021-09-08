"""
Quantifying VOI using some baselines
"""
import logging

import numpy as np
from scipy.stats import entropy

from pup.common.datatypes import Traj
from pup.common.grid import Grid
from pup.common.grid import create_grid_for_area
from pup.config import Config
from pup.io import preprocess

logger = logging.getLogger(__name__)


def quantify_voi_single_trajectory_baselines(traj_idx: int, trajectory: Traj, is_noisy_traj: bool):
    """
    Quantifying VOI for a single trajectory with baselines:
        - Size: number of checkins
        - Duration: time difference between the first and last checkins (in seconds)
        - Distance: sum of Euclidean distances between each consecutive points
        - Entropy: histogram entropy based on the provided grid

    :param traj_idx: trajectory index
    :param trajectory: trajectory
    :param grid: grid
    :return:
        - traj_index - trajectory index<br>
        - scores - scores as tuple of (size, duration, distance, entropy, previous_purchase_path_as_str)<br>
    """
    previous_purchase_path = 'NONE'

    size = len(trajectory)

    duration = trajectory[-1].timestamp - trajectory[0].timestamp

    total_distance = 0.0
    for i in range(1, len(trajectory)):
        c1 = trajectory[i-1]
        c2 = trajectory[i]
        total_distance += cal_euclidean_distance(c1.x, c1.y, c2.x, c2.y)

    grid_10 = create_grid_for_area(
        Config.eval_area_code,
        10,
        10,
        Config.eval_grid_boundary_order
    )

    hist_entropy_10 = cal_entropy(cal_histogram_for_trajectory(trajectory, grid_10).flat) if not is_noisy_traj else -1

    grid_100 = create_grid_for_area(
        Config.eval_area_code,
        100,
        100,
        Config.eval_grid_boundary_order
    )

    hist_entropy_100 = cal_entropy(cal_histogram_for_trajectory(trajectory, grid_100).flat) if not is_noisy_traj else -1

    grid_500 = create_grid_for_area(
        Config.eval_area_code,
        500,
        500,
        Config.eval_grid_boundary_order
    )

    hist_entropy_500 = cal_entropy(cal_histogram_for_trajectory(trajectory, grid_500).flat) if not is_noisy_traj else -1

    grid_1000 = create_grid_for_area(
        Config.eval_area_code,
        1000,
        1000,
        Config.eval_grid_boundary_order
    )

    hist_entropy_1000 = cal_entropy(cal_histogram_for_trajectory(trajectory, grid_1000).flat) if not is_noisy_traj else -1

    grid_2000 = create_grid_for_area(
        Config.eval_area_code,
        2000,
        2000,
        Config.eval_grid_boundary_order
    )

    hist_entropy_2000 = cal_entropy(cal_histogram_for_trajectory(trajectory, grid_2000).flat) if not is_noisy_traj else -1

    temporal_entropy_minute = cal_entropy(cal_temporal_histogram_for_trajectory(trajectory, width_in_minute=1)) if not is_noisy_traj else -1

    temporal_entropy_10minute = cal_entropy(cal_temporal_histogram_for_trajectory(trajectory, width_in_minute=10)) if not is_noisy_traj else -1

    max_gap = preprocess.find_max_gap(trajectory)

    scores = size, duration, total_distance, \
             hist_entropy_10, hist_entropy_100, hist_entropy_500, hist_entropy_1000, hist_entropy_2000, \
             temporal_entropy_minute, temporal_entropy_10minute, max_gap, previous_purchase_path

    return traj_idx, scores


def cal_euclidean_distance(x1, y1, x2, y2):
    """ Calculate Euclidean distance between (x1, y1) and (x2, y2)
    """
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))


def cal_entropy(a: np.ndarray, base=2) -> float:
    """
    Calculate entropy of a histogram

    Parameters
    ----------
    a
        the array storing values of the histogram
    base
        the base of logarithm, default is 2

    Returns
    -------
    float
        entropy
    """
    if np.isclose(np.sum(a), 0):
        return 0

    return entropy(a, base=base)


def cal_histogram_for_trajectory(traj: Traj, grid: Grid) -> np.ndarray:
    """
    Calculate histogram for locations of checkins in a trajectory within a grid.
    Any out of bound locations is projected to the border cells of the grid.

    Parameters
    ----------
    traj
        trajectory
    grid
        grid

    Returns
    -------
    numpy.ndarray
        histogram as an array
    """
    hist = np.zeros(grid.get_shape())

    for c in traj:
        x_idx, y_idx = grid.find_grid_index(c.x, c.y)
        hist[x_idx][y_idx] += 1

    return hist


def cal_temporal_histogram_for_trajectory(traj: Traj, width_in_minute=1) -> np.ndarray:
    """
    Calculate temporal histogram for locations of checkins in a trajectory where each cell is 1 minute.
    The first cell is the minute of the first checkin.

    Parameters
    ----------
    traj
        trajectory

    Returns
    -------
    numpy.ndarray
        temporal histogram as an array
    """
    # Find the start and end
    first_dt = traj[0].datetime.replace(second=0, microsecond=0)
    minute_count_dict = dict()
    for c in traj:
        dt = c.datetime.replace(second=0, microsecond=0)
        diff_minutes = int((dt - first_dt).total_seconds() / 60) // width_in_minute
        if diff_minutes not in minute_count_dict:
            minute_count_dict[diff_minutes] = 1
        else:
            minute_count_dict[diff_minutes] += 1

    hist_size = max(minute_count_dict) + 1

    hist = np.zeros(hist_size)

    for minute, count in minute_count_dict.items():
        hist[minute] = count

    return hist