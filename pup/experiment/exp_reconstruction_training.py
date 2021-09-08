"""
Executing training of reconstruction models
"""
import logging
from datetime import datetime
from multiprocessing import Process
from typing import List, Tuple

from pup.common import utils
from pup.common.datatypes import TrajList
from pup.common.enums import StartPriorType
from pup.common.grid import create_grid_for_area, Grid
from pup.config import Config
from pup.io import dataio, resultio
from pup.reconstruction import reconstruction_trainer
from pup.reconstruction.gpr import model_gpflow
from pup.degradation import degradator
from pup.voi import voi_io, voi_utils

logger = logging.getLogger(__name__)


def exe_exp_reconstruction_training_multi_users():
    """
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

    for user_id in range(start_user_id, end_user_id):
        start_traj_idx = Config.start_traj_idx if user_id == start_user_id else 0

        # Move all stuff into another function so that garbage collection can collect all junk there
        is_no_data = exe_reconstruction_training_single_user(user_id, grid, start_traj_idx)
        if is_no_data:
            break


def exe_reconstruction_training_single_user(user_id: int, grid: Grid, start_traj_idx: int) -> bool:
    data = dataio.load_data_by_config(at_subdir_idx=user_id)
    if len(data) == 0:
        logger.info('No data for user {}. Stop.'.format(user_id))
        return True

    trajectories = data[user_id]
    logger.info('User {}, got {} trajectories'.format(user_id, len(trajectories)))

    s = datetime.now()

    exe_reconstruction_training_single_user_multi_trajectories(user_id, trajectories, grid, start_traj_idx)

    exe_time = datetime.now().timestamp() - s.timestamp()

    logger.info('User {}\t Total exe time = {}'.format(user_id, exe_time))


def exe_reconstruction_training_single_user_multi_trajectories(user_id, trajectories: TrajList, grid: Grid, start_traj_idx: int):
    """
    """
    trajectory_interval = Config.trajectory_interval

    degradation_type, degradation_value = voi_utils.get_degradation_from_config()

    finished_set = resultio.get_finished_user_trajectory_indexes(
        in_dir=Config.output_dir,
        trajectory_interval=trajectory_interval,
        query_pricing_type=Config.query_pricing_type,
        degradation_type=degradation_type,
        degradation_value=degradation_value,
        transformation_type=Config.query_transformation_type,
        start_prior=Config.query_start_prior,
        previous_purchases=Config.query_previous_purchases,
    )

    degradation_type, degradation_value = voi_utils.get_degradation_from_config()
    transformation_type = Config.query_transformation_type
    previous_purchases = Config.query_previous_purchases

    traj_range = range(start_traj_idx, len(trajectories))
    # traj_range = range(start_traj_idx, start_traj_idx + 1)

    s = datetime.now()
    # Run SEQUENTIAL
    for traj_idx in traj_range:
        trajectory = trajectories[traj_idx]
        if (user_id, traj_idx) in finished_set:
            logger.info('Skipping training for user id {}, traj idx {} because finished'.format(user_id, traj_idx))
            continue

        if voi_io.single_component_pricing_check_models_existed(user_id, traj_idx):
            logger.info('Model existed for user id {}, traj idx {}'.format(user_id, traj_idx))
            continue

        exe_reconstruction_training_single_user_single_trajectory_new_process(
            user_id, traj_idx, trajectory, grid,
            degradation_type, degradation_value, previous_purchases, transformation_type
        )

    # Total running time
    exe_time = datetime.now().timestamp() - s.timestamp()
    logger.info('Total exe time = {}'.format(exe_time))


def exe_reconstruction_training_single_user_single_trajectory_new_process(
        user_id, traj_idx, trajectory, grid: Grid,
        degradation_type, degradation_value, previous_purchases, transformation_type):
    """
    Tensorflow has memory leak: after we trained a model and save its parameters, it did not release everything.
    So we need to run training part for each trajectory in a separate process,
    so that once it is done and we kill the process, every is cleared.

    :param user_id:
    :param traj_idx:
    :param trajectory:
    :param grid:
    :param degradation_type:
    :param degradation_value:
    :param previous_purchases:
    :param transformation_type:
    :return:
    """
    st = datetime.now()

    p = Process(
        target=exe_reconstruction_training_single_user_single_trajectory,
        args=(user_id, traj_idx, trajectory, grid,
              degradation_type, degradation_value, previous_purchases, transformation_type)
    )
    p.start()
    p.join()

    est = datetime.now().timestamp() - st.timestamp()
    logger.info('Trained user id {}, traj idx {} in {} seconds'.format(user_id, traj_idx, est))


def exe_reconstruction_training_single_user_single_trajectory(
        user_id, traj_idx, trajectory, grid: Grid,
        degradation_type, degradation_value, previous_purchases, transformation_type):
    seed = Config.query_random_seed
    utils.fix_random_seed(seed)

    model_gpflow.set_up_gpu()

    kernel_stds, kernel_variance_trainable = prepare_starting_kernel_std(grid)

    queried_data, is_noisy_traj = degradator.degrade_trajectory_and_combine_with_prior(
        trajectory, degradation_type, degradation_value, previous_purchases
    )

    models = reconstruction_trainer.train_GP_model(user_id, traj_idx, queried_data, kernel_stds, kernel_variance_trainable)
    voi_io.single_component_pricing_save_models(
        user_id, traj_idx, models,
        degradation_type=degradation_type, degradation_value=degradation_value, previous_purchases=previous_purchases)


def prepare_starting_kernel_std(grid: Grid) -> Tuple[List[float], bool]:
    start_prior = Config.query_start_prior
    if start_prior == StartPriorType.UNIFORM_GRID or start_prior == StartPriorType.CENTERED_NORMAL:
        # Assuming that the start prior distribution indicates that 2 x standard deviation is within the region
        # then the starting kernel std is 1/4 of the dimension
        length_x = grid.max_x - grid.min_x
        length_y = grid.max_y - grid.min_y

        return [length_x / 4.0, length_y / 4.0], False

    else:
        raise ValueError('Unknown start prior type: {}'.format(start_prior))


