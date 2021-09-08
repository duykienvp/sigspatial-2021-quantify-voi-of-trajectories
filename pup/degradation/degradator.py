"""
Degrade a trajectory if needed
"""
from pup.common.datatypes import Traj
from pup.common.enums import DegradationType, PreviousPurchaseType
from pup.config import Config
from pup.voi import voi_utils


def degrade_trajectory(trajectory: Traj,
                       degradation_type: DegradationType,
                       degradation_value: float,
                       previous_purchases: PreviousPurchaseType):
    """
    Degrade a trajectory based on configuration.
    If the trajectory does not need to be degraded, the original trajectory will be returned

    :param trajectory: trajectory to degrade
    :param degradation_type: type of degradation
    :param degradation_value: value of degradation (e.g., noise magnitude for ADD_NOISE degradation)
    :param previous_purchases: previous purchased if any
    :return: a degraded trajectory, is_noisy_trajectory
    """
    degraded_data = trajectory  # Default is query_degradation_type == NONE

    if degradation_type == DegradationType.ADD_NOISE:
        random_seed = Config.query_random_seed

        # Change the seed if new noisy trajectory is combined with previously perturbed trajectory
        # because we want different noise
        if previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_300_COMBINED or \
                previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_400_COMBINED:
            random_seed += 100

        degraded_data = add_noise.add_noises_to_trajectory(trajectory, degradation_value, random_seed)

    elif degradation_type == DegradationType.SUBSAMPLING or \
            degradation_type == DegradationType.SUBSTART or \
            degradation_type == DegradationType.SUB_TIME:
        degraded_data = subsampling.subsample_trajectory(trajectory, degradation_type, degradation_value)

    is_noisy_traj = degradation_type == DegradationType.ADD_NOISE

    return degraded_data, is_noisy_traj


def degrade_trajectory_from_configuration(trajectory: Traj):
    """ Degrade a trajectory with configuration from Config

    :param trajectory:
    :return: queried data, is_noisy_trajectory
    """
    degradation_type, degradation_value = voi_utils.get_degradation_from_config()
    previous_purchases = Config.query_previous_purchases

    # Degrade data
    return degrade_trajectory(trajectory, degradation_type, degradation_value, previous_purchases)


def degrade_trajectory_and_combine_with_prior(
        trajectory: Traj, degradation_type: DegradationType, degradation_value: float,
        previous_purchases: PreviousPurchaseType):
    """
    Degrade a trajectory of some degradation.
    Then combine the queried version with some priors if needed (which is only for perturbation)

    :param trajectory:
    :param degradation_type:
    :param degradation_value:
    :param previous_purchases:
    :return: degraded and combined data
    """
    degraded_data, is_noisy_traj = degrade_trajectory(
        trajectory, degradation_type, degradation_value, previous_purchases
    )

    if previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_300_COMBINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_400_COMBINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_300_COMBINED_RETRAINED or \
            previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_400_COMBINED_RETRAINED:
        # These are types of priors with need to combine

        # We need to combine two noisy trajectories
        if previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_300_COMBINED or \
                previous_purchases == PreviousPurchaseType.SAME_TRAJ_NOISE_300_COMBINED_RETRAINED:
            prev_degraded_data, is_prev_noisy_traj = degrade_trajectory(
                trajectory, degradation_type, voi_utils.PREVIOUS_PURCHASES_NOISE_300, PreviousPurchaseType.NONE
            )
        else:
            prev_degraded_data, is_prev_noisy_traj = degrade_trajectory(
                trajectory, degradation_type, voi_utils.PREVIOUS_PURCHASES_NOISE_400, PreviousPurchaseType.NONE
            )
        degraded_data = voi_utils.combine_noisy_queried_data(degraded_data, prev_degraded_data, previous_purchases)

    return degraded_data, is_noisy_traj