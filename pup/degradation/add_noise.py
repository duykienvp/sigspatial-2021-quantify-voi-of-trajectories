"""
Add noise to trajectory
"""
import copy
import random
from typing import List, Tuple

from pup.common.datatypes import Traj


def add_noises_to_trajectory(trajectory: Traj, noise_magnitude: float, random_seed) -> Traj:
    """ Add noises to a trajectory

    :param trajectory: the trajectory to add noise to
    :param noise_magnitude: noise magnitude (i.e., standard deviation)
    :param random_seed: random seed for noise generation
    :return: noise-added data
    """
    noises = get_noises(len(trajectory), noise_magnitude, random_seed)

    noisy_data = list()

    for i in range(len(trajectory)):
        noise_0 = noises[i][0]
        noise_1 = noises[i][1]

        c2 = copy.deepcopy(trajectory[i])
        c2.x += noise_0
        c2.y += noise_1
        c2.measurement_std = noise_magnitude

        noisy_data.append(c2)

    return noisy_data


def get_noises(trajectory_len: int, noise_magnitude: float, random_seed: int) -> List[Tuple[float, float]]:
    """ Get `trajectory_len` noise values where each noise value is a tuple (noise_0, noise_1),
    noise_i is a Gaussian white noise generated from N(0, noise_magnitude^2).
    For reproducibility, one random generator is created to generate all noises.

    Parameters
    ----------
    trajectory_len
        trajectory length (i.e. number of data points)
    noise_magnitude
        noise magnitude (i.e., standard deviation)
    random_seed
        random seed

    Returns
    -------
    list
        list of noise tuples (noise_0, noise_1)
    """
    noises = list()

    rand = random.Random(random_seed)
    for i in range(trajectory_len):
        noise_0 = rand.gauss(0, noise_magnitude)
        noise_1 = rand.gauss(0, noise_magnitude)

        noises.append((noise_0, noise_1))

    return noises
