"""
Executing prediction using reconstruction models
"""
import logging
from datetime import datetime

from tensorflow.python.framework.errors_impl import ResourceExhaustedError, InvalidArgumentError

from pup.common import utils
from pup.common.datatypes import TrajList
from pup.common.grid import create_grid_for_area, Grid
from pup.config import Config
from pup.io import dataio, resultio
from pup.reconstruction import reconstruction_predictor
from pup.reconstruction.gpr import model_gpflow
from pup.voi import voi_io, voi_utils
from pup.voi.voi_utils import cal_min_sigma_preds

logger = logging.getLogger(__name__)


def exe_reconstruction_predict_multi_users():
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
        is_no_data = exe_reconstruction_predict_single_user(user_id, grid, start_traj_idx)
        if is_no_data:
            break


def exe_reconstruction_predict_single_user(user_id: int, grid: Grid, start_traj_idx: int) -> bool:
    data = dataio.load_data_by_config(at_subdir_idx=user_id)
    if len(data) == 0:
        logger.info('No data for user {}. Stop.'.format(user_id))
        return True

    trajectories = data[user_id]
    logger.info('User {}, got {} trajectories'.format(user_id, len(trajectories)))

    s = datetime.now()

    exe_reconstruction_predict_single_user_multi_trajectories(user_id, trajectories, grid, start_traj_idx)

    exe_time = datetime.now().timestamp() - s.timestamp()

    logger.info('User {}\t Total exe time = {}'.format(user_id, exe_time))


def exe_reconstruction_predict_single_user_multi_trajectories(user_id, trajectories: TrajList, grid: Grid, start_traj_idx: int):
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
            logger.info('Skipping for user id {}, traj idx {} because finished'.format(user_id, traj_idx))
            continue

        # Load predictions file to see if we did it before
        if voi_io.quantify_voi_has_predictions(
                user_id, traj_idx, degradation_type, degradation_value, previous_purchases):
            logger.info('Already predicted for user id {}, traj idx {}'.format(user_id, traj_idx))
            continue

        exe_reconstruction_predict_single_user_single_trajectory(
            user_id, traj_idx, trajectory, grid,
            degradation_type, degradation_value, previous_purchases, transformation_type)


    # Total running time
    exe_time = datetime.now().timestamp() - s.timestamp()
    logger.info('Total exe time = {}'.format(exe_time))


def exe_reconstruction_predict_single_user_single_trajectory(
        user_id, traj_idx, trajectory, grid,
        degradation_type, degradation_value, previous_purchases, transformation_type):
    seed = Config.query_random_seed
    utils.fix_random_seed(seed)

    model_gpflow.set_up_gpu()

    try:

        logger.info('Predicting...')

        st = datetime.now()

        predictions = reconstruction_predictor.reconstruction_predict(
            user_id, traj_idx, trajectory, grid, degradation_type, degradation_value, previous_purchases, transformation_type
        )

        if predictions is not None:
            y_preds, sigmas_preds = predictions
            min_sigmas_preds = cal_min_sigma_preds(sigmas_preds)

            voi_io.single_component_pricing_save_predictions(
                user_id, traj_idx, min_sigmas_preds, degradation_type, degradation_value, previous_purchases
            )

        est = datetime.now().timestamp() - st.timestamp()
        logger.info('Predicted user id {}, traj idx {}, degrade {}, degrade value {} in {} seconds'.format(
            user_id, traj_idx, degradation_type.name, degradation_value, est))

    except ResourceExhaustedError as err:
        logger.error("ResourceExhaustedError user {}, traj {}: ".format(user_id, traj_idx, err.message))
    except InvalidArgumentError as err:
        logger.error("InvalidArgumentError user {}, traj {}: {}, ".format(user_id, traj_idx, err.message))


