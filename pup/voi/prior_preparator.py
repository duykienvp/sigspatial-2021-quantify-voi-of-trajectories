"""
Prepare priors
"""
from typing import List

import numpy as np
from scipy.stats import uniform

from pup.common.datatypes import Traj
from pup.common.enums import StartPriorType, PricingType
from pup.common.grid import Grid
from pup.config import Config
from pup.common import information_gain
from pup.voi import voi_utils


def prepare_prior_entropies(trajectory: Traj, grid: Grid, pricing_type: PricingType, base=2) -> List[float]:
    """
    Prepare prior entropies for a single trajectory.
    Other configuration is from Config

    :param trajectory: trajectory
    :param grid: grid
    :param pricing_type: pricing type
    :return: list of prior entropies
    """
    # Prior entropy for each prediction will be this value
    prior_entropy = cal_prior_entropy(grid, base)

    # These are timestamps we will evaluate on
    eval_timestamps = voi_utils.prepare_reconstruction_evaluation_timestamps(trajectory, pricing_type)
    num_timestamps = len(eval_timestamps)

    # For each timestamp, we give a prior entropy as calculated above
    prior_entropies = [prior_entropy] * num_timestamps

    return prior_entropies


def cal_prior_entropy(grid: Grid, base: float):
    """
    Calculate the entropy of the prior for a grid.
    Other configuration is from Config

    :param grid: grid
    :param base: logarithm base
    :return: the entropy of the prior
    """
    start_prior = Config.query_start_prior
    if start_prior == StartPriorType.UNIFORM_GRID:
        prior_entropy_x = uniform(loc=grid.min_x, scale=(grid.max_x - grid.min_x)).entropy()
        prior_entropy_y = uniform(loc=grid.min_y, scale=(grid.max_y - grid.min_y)).entropy()

        return (prior_entropy_x + prior_entropy_y) / np.log(base)

    elif start_prior == StartPriorType.CENTERED_NORMAL:
        length_x = grid.max_x - grid.min_x
        length_y = grid.max_y - grid.min_y
        std_x = length_x / 4
        std_y = length_y / 4
        entropy_x = information_gain.calculate_differential_entropy_norm(sigma=std_x, base=base)
        entropy_y = information_gain.calculate_differential_entropy_norm(sigma=std_y, base=base)

        return entropy_x + entropy_y

    else:
        raise ValueError('Unknown start prior type: {}'.format(start_prior))

