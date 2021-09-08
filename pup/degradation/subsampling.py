"""
Trajectory subsampling
"""
import random
from typing import List

from pup.common.datatypes import Traj
from pup.common.enums import DegradationType
from pup.config import Config


def subsample_trajectory(trajectory: Traj, degradation_type: DegradationType, subsampling_ratio: float) -> Traj:
    """
    Subsample a trajectory
    :param trajectory: trajectory to subsample
    :param degradation_type: type of subsampling
    :param subsampling_ratio: subsampling ratio
    :return: subsampled trajectory
    """
    selected_indexes = select_subsample_indexes(trajectory, degradation_type, subsampling_ratio)
    subsampled_data = [trajectory[i] for i in selected_indexes]

    return subsampled_data


def select_subsample_indexes(trajectory: Traj, degradation_type: DegradationType, subsampling_ratio) -> List[int]:
    """ Subsample a trajectory approximately to a ratio.
    Given the same length, ratio and seed, this function returns the same selected indexes.
    Given the same length and seed but 2 different ratios,
    the selected indexes of the larger ratio would include the selected indexes of the smaller ratio as a subset.

    :param trajectory: trajectory to subsample
    :param degradation_type: type of subsampling
    :param subsampling_ratio: subsampling ratio
    :return: sorted list of selected indexes
    """
    random_seed = Config.query_random_seed

    subsampling_ratio = max(0.0, subsampling_ratio)
    subsampling_ratio = min(subsampling_ratio, 1.0)

    trajectory_len = len(trajectory)

    num_points_kept = int(trajectory_len * subsampling_ratio)
    if num_points_kept < 1:
        num_points_kept = 1

    if degradation_type == DegradationType.SUBSAMPLING:
        # Random subsampling
        rand = random.Random(random_seed)

        # This implementation is wrong because it does not guarantee that with the same seed,
        # the selected values from a larger num_points always include the selected values from a smaller num_points
        # return sorted(rand.sample(list(range(trajectory_len)), num_points_kept))

        # This implementation guarantee it
        tmp = list(range(trajectory_len))
        rand.shuffle(tmp)
        return sorted(tmp[:num_points_kept])

    elif degradation_type == DegradationType.SUBSTART:
        # Cut from start
        return list(range(num_points_kept))

    elif degradation_type == DegradationType.SUB_TIME:
        # Uniformly randomly select x% of timestamps, if no timestamp is sampled, return the first data point
        start_timestamp = int(trajectory[0].timestamp)
        end_timestamp = int(trajectory[-1].timestamp)
        duration = end_timestamp - start_timestamp + 1

        num_timestamp_kept = int(duration * subsampling_ratio)
        if num_timestamp_kept < 1:
            num_timestamp_kept = 1

        rand = random.Random(random_seed)
        timestamps_kept = set(rand.sample(list(range(start_timestamp, end_timestamp + 1)), num_timestamp_kept))

        indexes_kept = list()
        for i in range(len(trajectory)):
            if int(trajectory[i].timestamp) in timestamps_kept:
                indexes_kept.append(i)

        if len(indexes_kept) == 0:
            indexes_kept.append(0)

        return indexes_kept

    else:
        raise ValueError('Invalid subsampling degradation type: {}'.format(degradation_type))

