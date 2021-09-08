from datetime import datetime

import numpy as np

from pup.common.constants import NUM_SECONDS_IN_DAY
from pup.common.datatypes import Traj, Checkin
from pup.common.enums import PricingType, DegradationType, ReconstructionMethod, TrajectoryIntervalType, \
    PreviousPurchaseType
from pup.config import Config
from pup.reconstruction import reconstruction_common

MIN_TRAJECTORY_SIZE_TO_SAVE_PREDICTIONS = 2000

PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_001 = 0.01
PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_005 = 0.05
PREVIOUS_PURCHASES_SUBSAMPLING_RATIO_02 = 0.2
PREVIOUS_PURCHASES_NOISE_300 = 300
PREVIOUS_PURCHASES_NOISE_400 = 400


def get_degradation_from_config():
    degradation_type = Config.query_degradation_type
    if degradation_type is None:
        raise ValueError('Invalid degradation type: {}'.format(degradation_type))

    degradation_value = 0.0

    if degradation_type == DegradationType.ADD_NOISE:
        degradation_value = Config.query_add_noise_magnitude

    elif degradation_type == DegradationType.SUBSAMPLING or \
            degradation_type == DegradationType.SUBSTART or \
            degradation_type == DegradationType.SUB_TIME:
        degradation_value = Config.query_subsampling_ratio

    return degradation_type, degradation_value


def prepare_reconstruction_evaluation_timestamps(trajectory: Traj, pricing_type: PricingType) -> list:
    """
    Prepare timestamps for reconstruction evaluation

    :param trajectory: the original trajectory
    :param pricing_type: pricing type
    :return: list of timestamps
    """
    if pricing_type == PricingType.IG_TRAJ_DURATION:
        start_timestamp = trajectory[0].timestamp
        end_timestamp = trajectory[-1].timestamp + 1
    elif pricing_type == PricingType.IG_TRAJ_DAY:
        t = datetime.fromtimestamp(trajectory[0].timestamp, trajectory[0].datetime.tzinfo)
        traj_date = datetime(t.year, t.month, t.day, 0, 0, 0, tzinfo=t.tzinfo)

        start_timestamp = traj_date.timestamp()
        end_timestamp = start_timestamp + NUM_SECONDS_IN_DAY
    else:
        raise ValueError("Not supported pricing type: {}".format(pricing_type.name))

    return list(range(int(start_timestamp), int(end_timestamp)))


def get_single_component_output_file_name(
        prefix, suffix,
        trajectory_interval, query_pricing_type,
        degradation_type, degradation_value,
        transformation_type, start_prior, previous_purchases,
        grid_cell_len=1000, default_location_measurement_std=3,
        reconstruction_method=ReconstructionMethod.GAUSSIAN_PROCESS) -> str:
    """
    Get the file name for result output of Single Component pricing

    :param prefix: file name prefix
    :param suffix: file name suffix
    :return: file name
    """

    output = '{}_grid_{}_defstd_{}_{}_pricing_{}_degrade_{}_trans_{}'.format(
        prefix,
        int(grid_cell_len),
        int(default_location_measurement_std),
        trajectory_interval.name,
        query_pricing_type.name,
        degradation_type.name,
        transformation_type.name
    )

    if degradation_type == DegradationType.SUBSAMPLING or \
            degradation_type == DegradationType.SUBSTART or \
            degradation_type == DegradationType.SUB_TIME:
        output = '{}_sub_ratio_{:.3f}'.format(output, degradation_value)
    elif degradation_type == DegradationType.ADD_NOISE:
        output = '{}_noise_{}'.format(output, int(degradation_value))
    elif degradation_type == DegradationType.NONE:
        output = '{}_no_degrade'.format(output)

    if query_pricing_type == PricingType.RECONSTRUCTION:
        output = '{}_reconstruct_{}'.format(
            output,
            reconstruction_method.name,
        )
    elif query_pricing_type == PricingType.IG_TRAJ_DAY or \
            query_pricing_type == PricingType.IG_TRAJ_DURATION:
        output = '{}_reconstruct_{}'.format(
            output,
            reconstruction_method.name,
        )
        output = '{}_prior_{}_prev_purchases_{}'.format(
            output,
            start_prior.name,
            previous_purchases.name
        )
    elif query_pricing_type == PricingType.HISTOGRAM_ENTROPY:
        output = '{}_hist_entropy'.format(
            output
        )

    if suffix is not None:
        output += '_{}'.format(suffix)

    return output


def combine_noisy_queried_data(noisy_traj: Traj, prev_noisy_traj: Traj, previous_purchases: PreviousPurchaseType) -> Traj:
    combined_traj: Traj = list()
    for i in range(len(noisy_traj)):
        new_c = noisy_traj[i]
        prev_c = prev_noisy_traj[i]

        # Combine mean


        # Combine previous purchase std with new std.
        # For example, if prev purchase is 300m, new purchase is 400m, combined noise = 1 /(1/(300^2) + 1/(400^2)) = 240
        if previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_300_COMBINED:
            prev_std = PREVIOUS_PURCHASES_NOISE_300
        else:
            prev_std = PREVIOUS_PURCHASES_NOISE_400

        new_std = reconstruction_common.prepare_measurement_std(noisy_traj)
        combined_std = combine_two_measurement_stds(prev_std, new_std)

        c = Checkin(c_id=new_c.c_id,
                    user_id=new_c.user_id,
                    timestamp=new_c.timestamp,
                    datetime=new_c.datetime,
                    lat=new_c.lat,
                    lon=new_c.lat,
                    measurement_std=new_c.measurement_std,
                    location_id=new_c.location_id,
                    trajectory_idx=new_c.trajectory_idx)

        c.x = combine_two_measurement_mean(new_c.x, prev_c.x, new_std, prev_std)
        c.y = combine_two_measurement_mean(new_c.y, prev_c.y, new_std, prev_std)
        c.measurement_std = combined_std

        combined_traj.append(c)

    return combined_traj


def combine_two_measurement_mean(new_mean: float, prev_mean: float, new_std: float, prev_std: float):
    """ Combine two measurement means using inverse-variance weighting

    Source: https://en.wikipedia.org/wiki/Inverse-variance_weighting

    :return:
    """
    new_w = 1 / (new_std * new_std)
    prev_w = 1 / (prev_std * prev_std)

    combined_mean = (new_w * new_mean + prev_w * prev_mean) / (new_w + prev_w)
    return combined_mean


def combine_two_measurement_stds(prev_std: float, new_std: float) -> float:
    """ Combine two measurement std using inverse-variance weighting

    Source: https://en.wikipedia.org/wiki/Inverse-variance_weighting

    :param prev_std:
    :param new_std:
    :return:
    """
    return np.sqrt(1.0 / (1.0 / (prev_std * prev_std) + 1.0 / (new_std * new_std)))


def cal_min_sigma_preds(sigmas_preds):
    """
    Calculate min of sigma for each predictions

    :param sigmas_preds: list of standard deviations for all predictions of all models
    :return: min sigmas in the same numpy size as the first input sigma
    """
    # find min sigmas
    min_sigmas_preds = sigmas_preds[0].copy()
    for i in range(1, len(sigmas_preds)):
        min_sigmas_preds = np.minimum(min_sigmas_preds, sigmas_preds[i])
    return min_sigmas_preds


def prepare_x_pred(trajectory: Traj, pricing_type: PricingType) -> np.ndarray:
    """
    Prepare x_pred

    :param trajectory: the original trajectory
    :param pricing_type: pricing type
    :return: x_pred as unscaled feature matrix of size n x 1
    """
    eval_timestamps = prepare_reconstruction_evaluation_timestamps(trajectory, pricing_type)

    x_values = np.asarray(eval_timestamps)
    x_values = x_values.reshape((-1, 1))

    return x_values