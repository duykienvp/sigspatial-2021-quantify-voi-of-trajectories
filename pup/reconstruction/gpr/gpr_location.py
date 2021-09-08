"""
Gaussian Processes for location data
"""
import logging
from typing import List

import numpy as np
from sklearn import preprocessing

from pup.config import Config
from pup.reconstruction.gpr import model_gpflow

logger = logging.getLogger(__name__)


def fit_unsplit_unscaled(x_train, y_train, measurement_std, kernel_type, kernel_stds: List[float], kernel_variance_trainable: bool, framework_type):
    """
    Fit using Gaussian Process Regressor for unsplit, unscaled input.

    We split input data into `max_training_size` chunks with each chunk overlapping `overlapping_size` points
    with the previous and next chunks. We then train each chunk separately.

    :param x_train: unscaled feature matrix of size n x 1
    :param y_train: unscaled label matrix of size n x 2
    :param measurement_std: default standard deviation of location measurement
    :param kernel_type: type of kernel
    :param framework_type: type of framework to use for GPs
    :return: list of trained models for each chunk
    """
    max_training_size = Config.reconstruction_max_training_size
    overlapping_size = Config.reconstruction_overlapping_size
    chunks = split_to_chunks(x_train, y_train, max_training_size, overlapping_size)
    logger.info('{} chunks'.format(len(chunks)))
    models = list()
    for x, y in chunks:
        gp_0, gp_1, scaler_x, scaler_y = fit_split_unscaled(x, y, measurement_std, kernel_type, kernel_stds, kernel_variance_trainable, framework_type)
        model = gp_0, gp_1, scaler_x, scaler_y
        models.append(model)

    return models


def fit_split_unscaled(x_train, y_train, measurement_std, kernel_type, kernel_stds: List[float], kernel_variance_trainable: bool, framework_type):
    """
    Fit using Gaussian Process Regressor for unscaled input.

    :param x_train: unscaled feature matrix of size n x 1
    :param y_train: unscaled label matrix of size n x 2
    :param measurement_std: default standard deviation of location measurement
    :param kernel_type: type of kernel
    :param framework_type: type of framework to use for GPs
    :return:
    """
    # Scale
    scaler_x = preprocessing.StandardScaler().fit(x_train)
    scaler_y = preprocessing.StandardScaler().fit(y_train)

    x_train_scaled = scaler_x.transform(x_train)
    y_train_scaled = scaler_y.transform(y_train)
    measurement_std_scaled = np.array([measurement_std, measurement_std]) / scaler_y.scale_
    kernel_stds_scaled = np.array(kernel_stds) / scaler_y.scale_

    # these are predicted locations and standard deviations
    # X, y, X_pred, kernels, kernel_type, gp_use_mean_func, measurement_std, parallel
    gp_0, gp_1 = model_gpflow.construct_fit_model_gpflow(
        x_train_scaled, y_train_scaled,
        kernel_type, measurement_std_scaled,
        kernel_stds_scaled, kernel_variance_trainable
    )

    return gp_0, gp_1, scaler_x, scaler_y


def split_to_chunks(x, y, max_chunk_size, overlapping_size):
    """
    We split input data into `max_training_size` chunks with each chunk overlapping `overlapping_size` points
    with the previous and next chunks.
    :param x: unscaled feature matrix of size n x 1
    :param y: unscaled label matrix of size n x 2
    :param max_chunk_size: the max size of each chunks
    :param overlapping_size: the #points overlapping between each consecutive chunks
    :return: list of tuples where each tuple contains (x_i, y_i) of i-th chunk
    """
    chunks = list()

    n = len(x)
    i = 0
    while True:
        next_i = min(i + max_chunk_size, n)
        chunks.append((x[i:next_i], y[i:next_i]))
        if n <= next_i:
            break
        i = next_i - overlapping_size

    return chunks
