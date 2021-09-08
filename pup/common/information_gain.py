# Calculate Information Gain

import logging
import numpy as np


logger = logging.getLogger(__name__)


def calculate_differential_entropy_norm(sigma, base=2) -> float:
    """ Differential entropy of a normal distribution with standard deviation sigma is: 0.5log(2*pi*e*sigma*sigma)

    :param sigma: standard deviation of the normal distribution
    :param base: base of logarithm
    :return: the differential entropy
    """
    return (0.5 * np.log(2 * np.pi * np.e * sigma * sigma)) / np.log(base)
