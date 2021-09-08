"""
Quantifying VOI using Information Gain
"""
import logging
from typing import List

import numpy as np

from pup.common.datatypes import Traj
from pup.common.enums import PreviousPurchaseType, DegradationType, TransformationType
from pup.common.grid import Grid
from pup.config import Config
from pup.voi.voi_utils import cal_min_sigma_preds, prepare_x_pred
from pup.reconstruction import reconstruction_predictor
from pup.common import information_gain
from pup.voi import prior_preparator, voi_utils, voi_io

logger = logging.getLogger(__name__)


def quantify_voi_single_trajectory_info_gain(
        user_id, traj_idx: int, trajectory: Traj, grid: Grid,
        degradation_type: DegradationType, degradation_value: float,
        previous_purchases: PreviousPurchaseType,
        transformation_type: TransformationType):
    """
    Quantifying VOI for a single trajectory with Information Gain

    :param user_id: user id
    :param traj_idx: trajectory index
    :param trajectory: trajectory
    :param grid: grid
    :return:
        - traj_index - trajectory index<br>
        - scores - scores as tuple of (Information_gain, previous_purchase_path_as_str)<br>
    """
    pricing_type = Config.query_pricing_type

    # Previous purchases assume the buyer bought something (or NONE if assumed bought nothing, then use prior entropy)
    if Config.query_previous_purchases == PreviousPurchaseType.NONE:
        # There is no previous purchases, so we use a start prior for it
        logger.info('Using start prior')
        prior_entropies = prior_preparator.prepare_prior_entropies(trajectory, grid, pricing_type)
        prev_purchase_entropies = prior_entropies
        previous_purchase_path = 'NONE'
    else:
        prev_purchase_entropies, previous_purchase_path = prepare_prev_purchase_entropies(
            user_id, traj_idx, trajectory, grid, transformation_type
        )
        if prev_purchase_entropies is None:
            logger.error('No previous purchases found for type {}'.format(Config.query_previous_purchases))
            raise RuntimeError('No previous purchases found')

    pred_entropies = prepare_prediction_entropies(
        user_id, traj_idx, trajectory, grid,
        degradation_type, degradation_value, previous_purchases, transformation_type
    )
    if pred_entropies is None:
        scores = (np.nan, previous_purchase_path)
        return traj_idx, scores

    info_gains = cal_information_gains(prev_purchase_entropies, pred_entropies)

    x_pred = prepare_x_pred(trajectory, pricing_type)
    total_info_gain = integrate_information_gains(x_pred, info_gains)

    # queried_data = querier.query_trajectory_from_configuration(trajectory, grid)

    scores = (total_info_gain, previous_purchase_path)
    return traj_idx, scores


def integrate_information_gains(x_pred: np.ndarray, info_gains: List[float]) -> float:
    """
    Integrate the total information gain.
    We approximate the integral by summing up the area of trapezoids
    forming from information gains as sides and x distance as height

    :param x_pred: x values
    :param info_gains: information gain for each x value
    :return: the total integrated information gain
    """
    integral = 0.0
    for i in range(1, len(info_gains)):
        prev_x = x_pred[i-1][0]
        x = x_pred[i][0]

        height = abs(x - prev_x)
        area = (info_gains[i-1] + info_gains[i]) * height / 2.0

        integral += area

    return integral


def cal_information_gains(prior_entropies, pred_entropies) -> List[float]:
    """
    Calculate information gains from prior and prediction entropies

    :param prior_entropies: prior entropies
    :param pred_entropies: prediction entropies
    :return: list of information gain for each entry
    """
    info_gains = list()
    for i in range(len(pred_entropies)):
        info_gain = prior_entropies[i] - pred_entropies[i]
        info_gains.append(info_gain)

    return info_gains


def prepare_prediction_entropies(user_id, traj_idx, trajectory, grid,
                                 degradation_type, degradation_value, previous_purchases,
                                 transformation_type):
    predictions = voi_io.voi_load_predictions(
        user_id, traj_idx, degradation_type, degradation_value, previous_purchases)
    if predictions is None:
        # There is no prediction, so we need to predict
        predictions = reconstruction_predictor.reconstruction_predict(
            user_id, traj_idx, trajectory, grid,
            degradation_type, degradation_value, previous_purchases, transformation_type
        )
        if predictions is None:
            raise RuntimeError('Error calculating predictions')

        _, sigmas_preds = predictions
        min_sigmas_preds = cal_min_sigma_preds(sigmas_preds)

        if voi_utils.MIN_TRAJECTORY_SIZE_TO_SAVE_PREDICTIONS < len(trajectory):
            # Save predictions of large trajectories
            voi_io.single_component_pricing_save_predictions(
                user_id, traj_idx, min_sigmas_preds, degradation_type, degradation_value, previous_purchases
            )

    else:
        # only sigmas preds are saved for last predictions
        logger.info('Loaded predictions')
        min_sigmas_preds = predictions

    pred_entropies = cal_prediction_entropies(min_sigmas_preds)

    return pred_entropies


def cal_prediction_entropies(sigmas_preds):
    """
    Calculate entropies of Gaussian predictions

    :param sigmas_preds: sigmas for predictions
    :return: list of prediction entropies of each x predict
    """
    pred_entropies = list()
    for i in range(len(sigmas_preds)):
        pred_0 = information_gain.calculate_differential_entropy_norm(sigma=sigmas_preds[i][0])
        pred_1 = information_gain.calculate_differential_entropy_norm(sigma=sigmas_preds[i][1])
        pred_entropies.append(pred_0 + pred_1)

    return pred_entropies


def prepare_prev_purchase_entropies(user_id, traj_idx, trajectory, grid, transformation_type):
    # Previous purchases assume the buyer bought something (or NONE if assumed bought nothing, then use prior entropy)
    prev_purchase_entropies = None
    previous_purchase_path = 'NONE'

    prev_purchase_type = Config.query_previous_purchases

    if prev_purchase_type == PreviousPurchaseType.NONE:
        # Buyer bought nothing before
        prev_purchase_entropies = None
        previous_purchase_path = 'NONE'

    elif prev_purchase_type == PreviousPurchaseType.FIRST_TRAJ:
        # Buyer bought the first trajectory, so we need to calculate the entropies for it
        pred_entropies = prepare_prediction_entropies(
            user_id=user_id, traj_idx=0,
            trajectory=trajectory, grid=grid,
            degradation_type=DegradationType.NONE, degradation_value=None,
            previous_purchases=PreviousPurchaseType.NONE,
            transformation_type=transformation_type
        )
        prev_purchase_entropies = pred_entropies
        previous_purchase_path = '({},{},{},{:.3f},{},{})'.format(
            user_id,
            traj_idx,
            Config.query_degradation_type.name,
            Config.query_subsampling_ratio,
            int(Config.query_add_noise_magnitude),
            Config.query_transformation_type.name
        )

    elif prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_NOISE_300 or \
            prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_NOISE_300_COMBINED or \
            prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_NOISE_300_COMBINED_RETRAINED:
        # Assume buyer bought the same trajectory but at noise 300m
        pred_entropies = prepare_prediction_entropies(
            user_id=user_id, traj_idx=traj_idx,
            trajectory=trajectory, grid=grid,
            degradation_type=DegradationType.ADD_NOISE,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_NOISE_300,
            previous_purchases=PreviousPurchaseType.NONE,
            transformation_type=transformation_type
        )
        prev_purchase_entropies = pred_entropies
        previous_purchase_path = '({},{},{},{:.3f},{},{})'.format(
            user_id,
            traj_idx,
            DegradationType.ADD_NOISE.name,
            -1,
            voi_utils.PREVIOUS_PURCHASES_NOISE_300,
            Config.query_transformation_type.name
        )

    elif prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_NOISE_400 or \
            prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_NOISE_400_COMBINED or \
            prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_NOISE_400_COMBINED_RETRAINED:
        # Assume buyer bought the same trajectory but at noise 400m
        pred_entropies = prepare_prediction_entropies(
            user_id=user_id, traj_idx=traj_idx,
            trajectory=trajectory, grid=grid,
            degradation_type=DegradationType.ADD_NOISE,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_NOISE_400,
            previous_purchases=PreviousPurchaseType.NONE,
            transformation_type=transformation_type
        )
        prev_purchase_entropies = pred_entropies
        previous_purchase_path = '({},{},{},{:.3f},{},{})'.format(
            user_id,
            traj_idx,
            DegradationType.ADD_NOISE.name,
            -1,
            voi_utils.PREVIOUS_PURCHASES_NOISE_400,
            Config.query_transformation_type.name
        )

    elif prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_001 or \
            prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_001_RETRAINED:
        # Assume buyer bought the same trajectory but at subsampling 1%
        pred_entropies = prepare_prediction_entropies(
            user_id=user_id, traj_idx=traj_idx,
            trajectory=trajectory, grid=grid,
            degradation_type=DegradationType.SUBSAMPLING,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_001,
            previous_purchases=PreviousPurchaseType.NONE,
            transformation_type=transformation_type
        )
        prev_purchase_entropies = pred_entropies
        previous_purchase_path = '({},{},{},{:.3f},{},{})'.format(
            user_id,
            traj_idx,
            DegradationType.SUBSAMPLING.name,
            voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_001,
            -1,
            Config.query_transformation_type.name
        )

    elif prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_005 or \
            prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_005_RETRAINED:
        # Assume buyer bought the same trajectory but at subsampling 5%
        pred_entropies = prepare_prediction_entropies(
            user_id=user_id, traj_idx=traj_idx,
            trajectory=trajectory, grid=grid,
            degradation_type=DegradationType.SUBSAMPLING,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_005,
            previous_purchases=PreviousPurchaseType.NONE,
            transformation_type=transformation_type
        )
        prev_purchase_entropies = pred_entropies
        previous_purchase_path = '({},{},{},{:.3f},{},{})'.format(
            user_id,
            traj_idx,
            DegradationType.SUBSAMPLING.name,
            voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_005,
            -1,
            Config.query_transformation_type.name
        )

    elif prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_02 or \
            prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_02_RETRAINED:
        # Assume buyer bought the same trajectory but at subsampling 20%
        pred_entropies = prepare_prediction_entropies(
            user_id=user_id, traj_idx=traj_idx,
            trajectory=trajectory, grid=grid,
            degradation_type=DegradationType.SUBSAMPLING,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_02,
            previous_purchases=PreviousPurchaseType.NONE,
            transformation_type=transformation_type
        )
        prev_purchase_entropies = pred_entropies
        previous_purchase_path = '({},{},{},{:.3f},{},{})'.format(
            user_id,
            traj_idx,
            DegradationType.SUBSAMPLING.name,
            voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_02,
            -1,
            Config.query_transformation_type.name
        )

    elif prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_START_001 or \
            prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_START_001_RETRAINED:
        # Assume buyer bought the same trajectory but at subsampling 1%
        pred_entropies = prepare_prediction_entropies(
            user_id=user_id, traj_idx=traj_idx,
            trajectory=trajectory, grid=grid,
            degradation_type=DegradationType.SUBSTART,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_001,
            previous_purchases=PreviousPurchaseType.NONE,
            transformation_type=transformation_type
        )
        prev_purchase_entropies = pred_entropies
        previous_purchase_path = '({},{},{},{:.3f},{},{})'.format(
            user_id,
            traj_idx,
            DegradationType.SUBSTART.name,
            voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_001,
            -1,
            Config.query_transformation_type.name
        )

    elif prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_START_005 or \
            prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_START_005_RETRAINED:
        # Assume buyer bought the same trajectory but at subsampling 5%
        pred_entropies = prepare_prediction_entropies(
            user_id=user_id, traj_idx=traj_idx,
            trajectory=trajectory, grid=grid,
            degradation_type=DegradationType.SUBSTART,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_005,
            previous_purchases=PreviousPurchaseType.NONE,
            transformation_type=transformation_type
        )
        prev_purchase_entropies = pred_entropies
        previous_purchase_path = '({},{},{},{:.3f},{},{})'.format(
            user_id,
            traj_idx,
            DegradationType.SUBSTART.name,
            voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_005,
            -1,
            Config.query_transformation_type.name
        )

    elif prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_START_02 or \
            prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_START_02_RETRAINED:
        # Assume buyer bought the same trajectory but at subsampling 20%
        pred_entropies = prepare_prediction_entropies(
            user_id=user_id, traj_idx=traj_idx,
            trajectory=trajectory, grid=grid,
            degradation_type=DegradationType.SUBSTART,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_02,
            previous_purchases=PreviousPurchaseType.NONE,
            transformation_type=transformation_type
        )
        prev_purchase_entropies = pred_entropies
        previous_purchase_path = '({},{},{},{:.3f},{},{})'.format(
            user_id,
            traj_idx,
            DegradationType.SUBSTART.name,
            voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_02,
            -1,
            Config.query_transformation_type.name
        )

    elif prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_TIME_001:
        # Assume buyer bought the same trajectory but at subsampling 1%
        pred_entropies = prepare_prediction_entropies(
            user_id=user_id, traj_idx=traj_idx,
            trajectory=trajectory, grid=grid,
            degradation_type=DegradationType.SUB_TIME,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_001,
            previous_purchases=PreviousPurchaseType.NONE,
            transformation_type=transformation_type
        )
        prev_purchase_entropies = pred_entropies
        previous_purchase_path = '({},{},{},{:.3f},{},{})'.format(
            user_id,
            traj_idx,
            DegradationType.SUB_TIME.name,
            voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_001,
            -1,
            Config.query_transformation_type.name
        )

    elif prev_purchase_type == PreviousPurchaseType.SAME_TRAJ_SUB_TIME_02:
        # Assume buyer bought the same trajectory but at subsampling 20%
        pred_entropies = prepare_prediction_entropies(
            user_id=user_id, traj_idx=traj_idx,
            trajectory=trajectory, grid=grid,
            degradation_type=DegradationType.SUB_TIME,
            degradation_value=voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_02,
            previous_purchases=PreviousPurchaseType.NONE,
            transformation_type=transformation_type
        )
        prev_purchase_entropies = pred_entropies
        previous_purchase_path = '({},{},{},{:.3f},{},{})'.format(
            user_id,
            traj_idx,
            DegradationType.SUB_TIME.name,
            voi_utils.PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_02,
            -1,
            Config.query_transformation_type.name
        )

    return prev_purchase_entropies, previous_purchase_path
