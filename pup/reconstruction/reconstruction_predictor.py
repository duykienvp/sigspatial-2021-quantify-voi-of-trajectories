"""
Predict locations using trained models
"""
import numpy as np

from pup.degradation import degradator
import pup.voi.voi_utils
from pup.common.datatypes import Traj
from pup.common.enums import PreviousPurchaseType, DegradationType, TransformationType
from pup.common.grid import Grid
from pup.config import Config
from pup.reconstruction import reconstruction_trainer, reconstruction_common
from pup.reconstruction.gpr import model_gpflow
from pup.voi import voi_io, voi_utils


def reconstruction_predict(user_id: int, traj_idx: int, trajectory: Traj, grid: Grid,
                           degradation_type: DegradationType, degradation_value: float,
                           previous_purchases: PreviousPurchaseType,
                           transformation_type: TransformationType):
    """
    Run predictions.
    If there is no previous purchases, we load the model of the provided degradation.
    If there is a previous purchase, we load the model of the previous purchase.
    Then, we query data base on configuration and provide that data to the model.

    :param grid:
    :param trajectory:
    :param user_id: user id
    :param traj_idx: trajectory index
    :return:
        - y_preds - list of predicted matrices of all models for this trajectory <br>
        - sigmas_preds - list of predicted sigmas of all models for this trajectory
    """
    # Load previous purchase model
    if previous_purchases == PreviousPurchaseType.NONE or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_300_COMBINED_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_400_COMBINED_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_001_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_005_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_02_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_START_001_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_START_005_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_START_02_RETRAINED:
        # We need to use the model trained from this data, not model trained from the prior
        if previous_purchases == PreviousPurchaseType.NONE or \
                previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_300_COMBINED_RETRAINED or \
                previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_400_COMBINED_RETRAINED:
            models = voi_io.single_component_pricing_load_models(
                user_id, traj_idx,
                degradation_type=degradation_type,
                degradation_value=degradation_value,
                previous_purchases=previous_purchases)
        else:
            # For subsampling/substart, we use the model with no previous purchase
            models = voi_io.single_component_pricing_load_models(
                user_id, traj_idx,
                degradation_type=degradation_type,
                degradation_value=degradation_value,
                previous_purchases=PreviousPurchaseType.NONE)

    elif previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_300 or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_300_COMBINED:
        models = voi_io.single_component_pricing_load_models(
            user_id, traj_idx,
            degradation_type=DegradationType.ADD_NOISE,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_NOISE_300,
            previous_purchases=PreviousPurchaseType.NONE)

    elif previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_400 or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_400_COMBINED:
        models = voi_io.single_component_pricing_load_models(
            user_id, traj_idx,
            degradation_type=DegradationType.ADD_NOISE,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_NOISE_400,
            previous_purchases=PreviousPurchaseType.NONE)

    elif previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_001:
        models = voi_io.single_component_pricing_load_models(
            user_id, traj_idx,
            degradation_type=DegradationType.SUBSAMPLING,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_001,
            previous_purchases=PreviousPurchaseType.NONE)

    elif previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_005:
        models = voi_io.single_component_pricing_load_models(
            user_id, traj_idx,
            degradation_type=DegradationType.SUBSAMPLING,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_005,
            previous_purchases=PreviousPurchaseType.NONE)

    elif previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_02:
        models = voi_io.single_component_pricing_load_models(
            user_id, traj_idx,
            degradation_type=DegradationType.SUBSAMPLING,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_02,
            previous_purchases=PreviousPurchaseType.NONE)

    elif previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_START_005:
        models = voi_io.single_component_pricing_load_models(
            user_id, traj_idx,
            degradation_type=DegradationType.SUBSTART,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_005,
            previous_purchases=PreviousPurchaseType.NONE)

    elif previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_START_001:
        models = voi_io.single_component_pricing_load_models(
            user_id, traj_idx,
            degradation_type=DegradationType.SUBSTART,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_001,
            previous_purchases=PreviousPurchaseType.NONE)

    elif previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_START_02:
        models = voi_io.single_component_pricing_load_models(
            user_id, traj_idx,
            degradation_type=DegradationType.SUBSTART,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_02,
            previous_purchases=PreviousPurchaseType.NONE)

    elif previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_TIME_001:
        models = voi_io.single_component_pricing_load_models(
            user_id, traj_idx,
            degradation_type=DegradationType.SUB_TIME,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_001,
            previous_purchases=PreviousPurchaseType.NONE)

    elif previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_TIME_02:
        models = voi_io.single_component_pricing_load_models(
            user_id, traj_idx,
            degradation_type=DegradationType.SUB_TIME,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_02,
            previous_purchases=PreviousPurchaseType.NONE)

    else:
        raise ValueError('Invalid previous purchase type: {}'.format(previous_purchases))

    if models is None:
        return None

    # Prepare standard deviation for data in case the previous purchase was noisy data
    queried_data, is_noisy_traj = degradator.degrade_trajectory_and_combine_with_prior(
        trajectory, degradation_type, degradation_value, previous_purchases
    )

    measurement_std = reconstruction_common.prepare_measurement_std(queried_data)
    chunks = split_data_to_chunks(
        queried_data,
        Config.reconstruction_max_training_size,
        Config.reconstruction_overlapping_size)

    # Predict
    pricing_type = Config.query_pricing_type
    x_pred = pup.voi.voi_utils.prepare_x_pred(trajectory, pricing_type)
    print('x_pred size = {}'.format(len(x_pred)))

    if previous_purchases == PreviousPurchaseType.NONE or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_300 or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_300_COMBINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_400 or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_400_COMBINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_300_COMBINED_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_400_COMBINED_RETRAINED:
        return predict_prev_nodegrade_or_noise(models, chunks, x_pred, measurement_std)

    elif previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_001 or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_005 or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_02 or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_001_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_005_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_02_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_START_001 or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_START_005 or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_START_02 or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_START_001_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_START_005_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_START_02_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_TIME_001 or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_SUB_TIME_02:
        return predict_prev_subsampling(models, chunks, x_pred, measurement_std)

    else:
        raise ValueError('Invalid previous purchases type: {}'.format(previous_purchases))


def predict_prev_nodegrade_or_noise(models, chunks, x_pred, measurement_std):
    """
    Prediction when previous purchases are either nothing or noise-added trajectories

    :param models:
    :param chunks:
    :param x_pred:
    :param measurement_std:
    :return: y_preds, sigmas_preds
    """
    y_preds = list()
    sigmas_preds = list()
    for i in range(len(models)):
        gp_0, gp_1, scaler_x, scaler_y = models[i]

        model = prepare_model(chunks[i], gp_0, gp_1, scaler_x, scaler_y, measurement_std)

        # Divide the x_pred in case size is too large
        x_pred_scaled = scaler_x.transform(x_pred)

        g_mean_full, g_std_full = pred(model, x_pred_scaled, scaler_y)

        y_preds.append(g_mean_full)
        sigmas_preds.append(g_std_full)

    return y_preds, sigmas_preds


def predict_prev_subsampling(models, chunks, x_pred, measurement_std):
    """
    Prediction when previous purchases are subsampled trajectories

    :param models:
    :param chunks:
    :param x_pred:
    :param measurement_std:
    :return: y_preds, sigmas_preds
    """
    y_preds = list()
    sigmas_preds = list()
    for i in range(len(models)):
        gp_0, gp_1, scaler_x, scaler_y = models[i]
        # Divide the x_pred in case size is too large
        x_pred_scaled = scaler_x.transform(x_pred)

        for chunk in chunks:
            model = prepare_model(chunk, gp_0, gp_1, scaler_x, scaler_y, measurement_std)
            g_mean_full, g_std_full = pred(model, x_pred_scaled, scaler_y)

            y_preds.append(g_mean_full)
            sigmas_preds.append(g_std_full)

    return y_preds, sigmas_preds


def split_data_to_chunks(data: list, max_chunk_size: int, overlapping_size: int):
    """
    Because GP can take very long to finish, we split data into smaller chunks and train/predict these chunks separately

    :param data:
    :param max_chunk_size:
    :param overlapping_size:
    :return: list of split data
    """
    chunks = list()
    n = len(data)
    i = 0
    while True:
        next_i = min(i + max_chunk_size, n)
        chunks.append(data[i:next_i])
        if n <= next_i:
            break
        i = next_i - overlapping_size

    return chunks


def prepare_model(queried_data, gp_0, gp_1, scaler_x, scaler_y, measurement_std):
    """
    Prepare a model for a set of data given previously trained parameters

    :param queried_data:
    :param gp_0:
    :param gp_1:
    :param scaler_x:
    :param scaler_y:
    :param measurement_std:
    :return:
    """
    # Prepare model with this data and previously trained parameters
    x_train = reconstruction_trainer.prepare_x_train(queried_data)
    y_train = reconstruction_trainer.prepare_y_train(queried_data)
    x_train_scaled = scaler_x.transform(x_train)
    y_train_scaled = scaler_y.transform(y_train)
    measurement_std_scaled = np.array([measurement_std, measurement_std]) / scaler_y.scale_

    X = x_train_scaled
    y = y_train_scaled

    kernel_type = Config.reconstruction_gp_kernel

    new_model_0 = model_gpflow.construct_model_for_new_data(
        X, y[:, 0].reshape((-1, 1)), gp_0, kernel_type, measurement_std_scaled[0]
    )
    new_model_1 = model_gpflow.construct_model_for_new_data(
        X, y[:, 1].reshape((-1, 1)), gp_1, kernel_type, measurement_std_scaled[1]
    )

    model = (new_model_0, new_model_1)
    return model


def pred(model, x_pred_scaled, scaler_y):
    """
    Predict

    :param model: model for prediction
    :param x_pred_scaled: scaled x values we need to predict for
    :param scaler_y: scaler for y values
    :return:
    """
    MAX_PREDICT_SIZE = 10000
    g_mean_full = g_std_full = None

    start = 0
    while start < len(x_pred_scaled):
        end = start + MAX_PREDICT_SIZE
        x_pred_scaled_slice = x_pred_scaled[start:end]

        g_mean_scaled, g_std_scaled = model_gpflow.predict_gpflow(model, x_pred_scaled_slice)

        g_mean = scaler_y.inverse_transform(g_mean_scaled)
        g_std = g_std_scaled * scaler_y.scale_

        if g_mean_full is None:
            g_mean_full = g_mean
            g_std_full = g_std
        else:
            g_mean_full = np.vstack((g_mean_full, g_mean))
            g_std_full = np.vstack((g_std_full, g_std))

        start = end

    return g_mean_full, g_std_full
