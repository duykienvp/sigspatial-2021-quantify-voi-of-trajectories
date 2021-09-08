"""
Executing quantifying VOI using IG
"""
import logging
import os
from datetime import datetime

import pup.io.dataio
from pup.common import utils
from pup.common.datatypes import TrajList, Traj
from pup.common.enums import PricingType
from pup.common.grid import create_grid_for_area, Grid
from pup.config import Config
from pup.io import dataio, resultio
from pup.reconstruction.gpr import model_gpflow
from pup.degradation import degradator
from pup.voi import voi_utils, voi_io, \
    voi_baselines, voi_info_gain

logger = logging.getLogger(__name__)


def exe_exp_quantify_voi_multi_users():
    """ Executing quantifying VOI for multiple users
    """
    grid = create_grid_for_area(
        Config.eval_area_code,
        Config.eval_grid_cell_len_x,
        Config.eval_grid_cell_len_y,
        Config.eval_grid_boundary_order
    )
    logger.info('Grid for area {}: {}'.format(Config.eval_area_code, grid))
    logger.info('Grid shape: {}'.format(grid.get_shape()))

    start_user_id = Config.start_user_id
    end_user_id = Config.end_user_id if Config.end_user_id is not None else 1000000000

    write_header()
    for user_id in range(start_user_id, end_user_id):
        start_traj_idx = Config.start_traj_idx if user_id == start_user_id else 0

        # Move all stuff into another function so that garbage collection can collect all junk there
        is_no_data = exe_quantify_voi_single_user(user_id, grid, start_traj_idx)
        if is_no_data:
            break


def exe_quantify_voi_single_user(user_id: int, grid: Grid, start_traj_idx: int) -> bool:
    """
    Executing quantifying VOI for single user at a starting trajectory index
    :param user_id:
    :param grid:
    :param start_traj_idx:
    :return:
    """
    data = dataio.load_data_by_config(at_subdir_idx=user_id)
    if len(data) == 0:
        logger.info('No data for user {}. Stop.'.format(user_id))
        return True

    trajectories = data[user_id]
    logger.info('User {}, got {} trajectories'.format(user_id, len(trajectories)))

    s = datetime.now()

    exe_quantify_voi_single_user_multi_trajectories(user_id, trajectories, grid, start_traj_idx)

    exe_time = datetime.now().timestamp() - s.timestamp()

    logger.info('User {}\t Total exe time = {}'.format(user_id, exe_time))


def exe_quantify_voi_single_user_multi_trajectories(user_id, trajectories: TrajList, grid: Grid, start_traj_idx: int):
    """
    Executing quantifying VOI multiple trajectories of a single user at a starting trajectory index
    """
    trajectory_interval = Config.trajectory_interval

    if Config.query_pricing_type != PricingType.BASELINES:
        degradation_type, degradation_value = voi_utils.get_degradation_from_config()

        finished = resultio.get_finished_user_trajectory_indexes(
            in_dir=Config.output_dir,
            trajectory_interval=trajectory_interval,
            query_pricing_type=Config.query_pricing_type,
            degradation_type=degradation_type,
            degradation_value=degradation_value,
            transformation_type=Config.query_transformation_type,
            start_prior=Config.query_start_prior,
            previous_purchases=Config.query_previous_purchases,
        )
    else:
        finished = set()

    traj_range = range(start_traj_idx, len(trajectories))
    # traj_range = range(start_traj_idx, start_traj_idx + 1)

    s = datetime.now()
    # Run SEQUENTIAL
    for traj_idx in traj_range:
        trajectory = trajectories[traj_idx]

        if Config.query_pricing_type != PricingType.BASELINES:
            if (user_id, traj_idx) in finished:
                logger.info('Skipping for user id {}, traj idx {} because finished'.format(user_id, traj_idx))
                continue

        st = datetime.now()

        traj_idx, scores = exe_quantify_voi_single_user_single_trajectory(user_id, traj_idx, trajectory, grid)

        est = datetime.now().timestamp() - st.timestamp()
        logger.info('Finished user id {}, traj idx {} in {} seconds'.format(user_id, traj_idx, est))

        traj_size = len(trajectory)
        traj_duration = trajectory[-1].timestamp - trajectory[0].timestamp
        write_exp_result(user_id, traj_idx, traj_size, traj_duration, scores, est)

    # Total running time
    exe_time = datetime.now().timestamp() - s.timestamp()
    logger.info('Total exe time = {}'.format(exe_time))


def exe_quantify_voi_single_user_single_trajectory(user_id, traj_idx: int, trajectory: Traj, grid: Grid):
    """
    Executing quantifying VOI of a trajectory of a single user

    :param user_id: user id
    :param traj_idx: trajectory index
    :param trajectory: trajectory
    :param grid: grid
    :return:
        - traj_index - trajectory index<br>
        - scores - scores as tuple of (Information_gain, previous_purchase_path_as_str)<br>
        - exe_time - execution time
    """
    pricing_type = Config.query_pricing_type

    if pricing_type == PricingType.BASELINES:
        queried_data, is_noisy_traj = degradator.degrade_trajectory_from_configuration(trajectory)

        return voi_baselines.quantify_voi_single_trajectory_baselines(
            traj_idx, queried_data, is_noisy_traj
        )

    if pricing_type == PricingType.IG_TRAJ_DURATION or pricing_type == PricingType.IG_TRAJ_DAY:
        return exe_quantify_voi_single_user_single_trajectory_info_gain(user_id, traj_idx, trajectory, grid)


def exe_quantify_voi_single_user_single_trajectory_info_gain(user_id, traj_idx, trajectory, grid):
    """
    Executing quantifying VOI of a trajectory of a single user using Info Gain
    :param user_id:
    :param traj_idx:
    :param trajectory:
    :param grid:
    :return:
    """
    seed = Config.query_random_seed
    utils.fix_random_seed(seed)

    model_gpflow.set_up_gpu()

    degradation_type, degradation_value = voi_utils.get_degradation_from_config()
    transformation_type = Config.query_transformation_type
    previous_purchases = Config.query_previous_purchases

    return voi_info_gain.quantify_voi_single_trajectory_info_gain(
        user_id, traj_idx, trajectory, grid,
        degradation_type, degradation_value, previous_purchases, transformation_type
    )


def write_header():
    """
    Write header of output file
    """
    header = [
        'time',
        'data_dir_name',
        'eval_area_code',
        'eval_grid_cell_len',
        'eval_grid_boundary_order',
        'default_measurement_std',
        'trajectory_interval',
        'query_pricing_type',
        'degradation_type',
        'transformation_type',
        'random_seed',
        'subsampling_ratio',
        'added_noise_magnitude',
        'user_id',
        'traj_index',
        'traj_size',
        'traj_duration'
    ]
    if Config.query_pricing_type == PricingType.RECONSTRUCTION:
        header.extend([
            'reconstruction_method',
            'reconstruction_gp_framework',
            'reconstruction_gp_kernel',
            'mean_kl_divergence',
            'median_kl_divergence',
            'rmse_kl_divergence',
            'mean_distances',
            'median_distances',
            'rmse_distances',
            'mean_energy_scores',
            'median_energy_scores',
            'rmse_energy_scores'
        ])
    elif Config.query_pricing_type == PricingType.IG_TRAJ_DAY or \
            Config.query_pricing_type == PricingType.IG_TRAJ_DURATION:
        header.extend([
            'reconstruction_method',
            'reconstruction_gp_framework',
            'reconstruction_gp_kernel',
            'start_prior',
            'previous_purchases',
            'previous_purchases_path',
            'total_info_gain'
        ])
    elif Config.query_pricing_type == PricingType.HISTOGRAM_ENTROPY:
        header.extend([
            'histogram_entropy'
        ])
    elif Config.query_pricing_type == PricingType.MARKOV_CHAIN_ENTROPY:
        header.extend([
            'mc_entropy'
        ])

    elif Config.query_pricing_type == PricingType.TRAVEL_DISTANCE:
        header.extend([
            'travel_distance'
        ])
    elif Config.query_pricing_type == PricingType.BASELINES:
        header.extend([
            'previous_purchase_path',
            'max_gap',
            'size',
            'duration',
            'total_distance',
            'hist_entropy_10',
            'hist_entropy_100',
            'hist_entropy_500',
            'hist_entropy_1000',
            'hist_entropy_2000',
            'temporal_entropy_minute',
            'temporal_entropy_10minute'
        ])

    header.append('exe_time')

    output_file = get_output_file()
    pup.io.dataio.write_line(output_file, '\t'.join(header))


def write_exp_result(user_id, traj_idx, traj_size, traj_duration, scores, exe_time):
    """
    Write experiment results for file

    :param user_id:
    :param traj_idx:
    :param traj_size:
    :param traj_duration:
    :param scores:
    :param exe_time:
    :return:
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_dir_name = os.path.basename(os.path.normpath(Config.data_dir))

    output = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.5f}\t{}\t{}\t{}\t{}\t{}'.format(
        current_time,
        data_dir_name,
        Config.eval_area_code,
        int(Config.eval_grid_cell_len_x),
        int(Config.eval_grid_boundary_order),
        int(Config.eval_default_location_measurement_std),
        Config.trajectory_interval.name,
        Config.query_pricing_type.name,
        Config.query_degradation_type.name,
        Config.query_transformation_type.name,
        Config.query_random_seed,
        Config.query_subsampling_ratio,
        Config.query_add_noise_magnitude,
        user_id,
        traj_idx,
        traj_size,
        traj_duration
    )

    if Config.query_pricing_type == PricingType.RECONSTRUCTION:
        (
            mean_kl_divergence,
            median_kl_divergence,
            rmse_kl_divergence,
            mean_distances,
            median_distances,
            rmse_distances,
            mean_energy_scores,
            median_energy_scores,
            rmse_energy_scores
        ) = scores

        reconstruct_output = '\t{}\t{}\t{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(
            Config.reconstruction_method.name,
            Config.reconstruction_gp_framework.name,
            Config.reconstruction_gp_kernel.name,
            mean_kl_divergence,
            median_kl_divergence,
            rmse_kl_divergence,
            mean_distances,
            median_distances,
            rmse_distances,
            mean_energy_scores,
            median_energy_scores,
            rmse_energy_scores,
        )
        output += reconstruct_output

    elif Config.query_pricing_type == PricingType.IG_TRAJ_DAY or \
            Config.query_pricing_type == PricingType.IG_TRAJ_DURATION:
        total_info_gain, previous_purchase_path = scores
        reconstruct_output = '\t{}\t{}\t{}\t{}\t{}\t{}\t{:.5f}'.format(
            Config.reconstruction_method.name,
            Config.reconstruction_gp_framework.name,
            Config.reconstruction_gp_kernel.name,
            Config.query_start_prior.name,
            Config.query_previous_purchases.name,
            previous_purchase_path,
            total_info_gain
        )
        output += reconstruct_output

    elif Config.query_pricing_type == PricingType.HISTOGRAM_ENTROPY:
        hist_entropy = scores
        output += '\t{:.5f}'.format(hist_entropy)

    elif Config.query_pricing_type == PricingType.MARKOV_CHAIN_ENTROPY:
        hist_entropy = scores
        output += '\t{:.5f}'.format(hist_entropy)

    elif Config.query_pricing_type == PricingType.TRAVEL_DISTANCE:
        travel_distance = scores
        output += '\t{:.2f}'.format(travel_distance)

    elif Config.query_pricing_type == PricingType.BASELINES:
        size, duration, total_distance, \
        hist_entropy_10, hist_entropy_100, hist_entropy_500, hist_entropy_1000, hist_entropy_2000, \
        temporal_entropy_minute, temporal_entropy_10minute, max_gap, previous_purchase_path = scores
        reconstruct_output = '\t{}\t{}\t{}\t{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(
            previous_purchase_path,
            max_gap,
            int(size),
            int(duration),
            total_distance,
            hist_entropy_10,
            hist_entropy_100,
            hist_entropy_500,
            hist_entropy_1000,
            hist_entropy_2000,
            temporal_entropy_minute,
            temporal_entropy_10minute
        )
        output += reconstruct_output

    output += '\t{:.5f}'.format(exe_time)

    print(output)

    output_file = get_output_file()
    pup.io.dataio.write_line(output_file, output)


def get_output_file():
    """
    Get the output file

    :return: output file
    """
    degradation_type, degradation_value = voi_utils.get_degradation_from_config()
    transformation_type = Config.query_transformation_type
    previous_purchases = Config.query_previous_purchases
    trajectory_interval = Config.trajectory_interval
    query_pricing_type = Config.query_pricing_type
    start_prior = Config.query_start_prior

    out_dir = os.path.join(Config.output_dir, voi_io.get_dir_name_for_degradation(None, None))
    out_file_prefix = '{}_{}_{}'.format(os.path.basename(Config.output_file), Config.start_user_id, Config.end_user_id)
    out_file_prefix = os.path.join(out_dir, out_file_prefix)

    output_file = voi_utils.get_single_component_output_file_name(
        out_file_prefix, '.csv', trajectory_interval, query_pricing_type,
        degradation_type, degradation_value,
        transformation_type, start_prior, previous_purchases
    )

    return output_file
