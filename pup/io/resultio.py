import logging
import os
from typing import Set, Tuple

import pandas as pd
import numpy as np

from pup.common.enums import DegradationType, ReconstructionMethod
from pup.voi import voi_io

logger = logging.getLogger(__name__)


def remove_results_duplicates(df, baselines=False, additional_cols=None):
    # filter duplicate rows based on specific columns:
    comparing_columns = [
        'eval_area_code',
        'default_measurement_std',
        'trajectory_interval',
        'query_pricing_type',
        'degradation_type',
        'transformation_type',
        'random_seed',
        'user_id',
        'traj_index'
    ]
    if additional_cols is not None:
        comparing_columns.extend(additional_cols)
    if baselines:
        return df.sort_values('size').drop_duplicates(subset=comparing_columns, keep='last')
    else:
        comparing_columns.extend([
            'reconstruction_method',
            'reconstruction_gp_framework',
            'reconstruction_gp_kernel',
            'start_prior',
            'previous_purchases'
        ])

        return df.sort_values('total_info_gain').drop_duplicates(subset=comparing_columns, keep='last')


def read_results(input_file):
    df = pd.read_table(input_file)

    try:
        # filter out the rows with `exe_time` value in corresponding column
        df = df[df.exe_time.str.contains('exe_time') == False]

    except AttributeError as err:
        pass

    df['eval_grid_cell_len'] = pd.to_numeric(df['eval_grid_cell_len'])
    df['eval_grid_boundary_order'] = pd.to_numeric(df['eval_grid_boundary_order'])
    df['default_measurement_std'] = pd.to_numeric(df['default_measurement_std'])
    df['random_seed'] = pd.to_numeric(df['random_seed'])
    df['subsampling_ratio'] = pd.to_numeric(df['subsampling_ratio'])
    df['added_noise_magnitude'] = pd.to_numeric(df['added_noise_magnitude'])
    df['user_id'] = pd.to_numeric(df['user_id'])
    df['traj_index'] = pd.to_numeric(df['traj_index'])
    df['traj_size'] = pd.to_numeric(df['traj_size'])
    df['traj_duration'] = pd.to_numeric(df['traj_duration'])
    df['exe_time'] = pd.to_numeric(df['exe_time'])
    df['total_info_gain'] = pd.to_numeric(df['total_info_gain'])

    df = remove_results_duplicates(df)

    return df


def list_files_in_dir(dir_path, prefix='output'):
    return [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and os.path.basename(f).startswith(prefix)
    ]


def read_finished_data(start_dir, degradation_type, degradation_value):
    degradation_dir = voi_io.get_dir_name_for_degradation(degradation_type, degradation_value)
    input_dir = os.path.join(start_dir, degradation_dir)
    files = list_files_in_dir(input_dir)
    if 0 < len(files):

        df = read_results(files[0])

        for i in range(1, len(files)):
            # print('Reading {}'.format(files[i]))
            df_i = read_results(files[i])

            df = df.append(df_i)
        return remove_results_duplicates(df)
    else:
        return None


def read_baselines_from_file(input_file):
    df = pd.read_table(input_file)

    try:
        # filter out the rows with `exe_time` value in corresponding column
        df = df[df.exe_time.str.contains('exe_time') == False]

    except AttributeError as err:
        pass

    df['eval_grid_cell_len'] = pd.to_numeric(df['eval_grid_cell_len'])
    df['eval_grid_boundary_order'] = pd.to_numeric(df['eval_grid_boundary_order'])
    df['default_measurement_std'] = pd.to_numeric(df['default_measurement_std'])
    df['random_seed'] = pd.to_numeric(df['random_seed'])
    df['subsampling_ratio'] = pd.to_numeric(df['subsampling_ratio'])
    df['added_noise_magnitude'] = pd.to_numeric(df['added_noise_magnitude'])
    df['user_id'] = pd.to_numeric(df['user_id'])
    df['traj_index'] = pd.to_numeric(df['traj_index'])
    df['traj_size'] = pd.to_numeric(df['traj_size'])
    df['traj_duration'] = pd.to_numeric(df['traj_duration'])
    df['exe_time'] = pd.to_numeric(df['exe_time'])

    df['max_gap'] = pd.to_numeric(df['max_gap'])
    df['size'] = pd.to_numeric(df['size'])
    df['duration'] = pd.to_numeric(df['duration'])
    df['total_distance'] = pd.to_numeric(df['total_distance'])
    df['hist_entropy_10'] = pd.to_numeric(df['hist_entropy_10'])
    df['hist_entropy_100'] = pd.to_numeric(df['hist_entropy_100'])
    df['hist_entropy_500'] = pd.to_numeric(df['hist_entropy_500'])
    df['hist_entropy_1000'] = pd.to_numeric(df['hist_entropy_1000'])
    df['hist_entropy_2000'] = pd.to_numeric(df['hist_entropy_2000'])

    if 'temporal_entropy_minute' in df.columns:
        df['temporal_entropy_minute'] = pd.to_numeric(df['temporal_entropy_minute'])

    df = remove_results_duplicates(df, baselines=True)

    return df


def read_baselines_for_degradation(start_dir, degradation_type, degradation_value):
    degradation_dir = voi_io.get_dir_name_for_degradation(degradation_type, degradation_value)
    input_dir = os.path.join(start_dir, degradation_dir)
    files = list_files_in_dir(input_dir)
    if 0 < len(files):

        df = read_baselines_from_file(files[0])

        for i in range(1, len(files)):
            # print('Reading {}'.format(files[i]))
            df_i = read_baselines_from_file(files[i])

            df = df.append(df_i)
        return remove_results_duplicates(df, baselines=True)
    else:
        return None


def read_baselines(baselines_dirs):

    baselines_df = read_baselines_for_degradation(baselines_dirs, DegradationType.NONE, None)

    # baselines_df = baselines_df.append(read_baselines_for_degradation(baselines_dirs, DegradationType.ADD_NOISE, 10))
    # baselines_df = baselines_df.append(read_baselines_for_degradation(baselines_dirs, DegradationType.ADD_NOISE, 100))
    # baselines_df = baselines_df.append(read_baselines_for_degradation(baselines_dirs, DegradationType.ADD_NOISE, 200))
    # baselines_df = baselines_df.append(read_baselines_for_degradation(baselines_dirs, DegradationType.ADD_NOISE, 300))
    # baselines_df = baselines_df.append(read_baselines_for_degradation(baselines_dirs, DegradationType.ADD_NOISE, 400))
    #
    # baselines_df = baselines_df.append(read_baselines_for_degradation(baselines_dirs, DegradationType.SUBSAMPLING, 0.2))
    # baselines_df = baselines_df.append(read_baselines_for_degradation(baselines_dirs, DegradationType.SUBSAMPLING, 0.4))
    # baselines_df = baselines_df.append(read_baselines_for_degradation(baselines_dirs, DegradationType.SUBSAMPLING, 0.6))
    # baselines_df = baselines_df.append(read_baselines_for_degradation(baselines_dirs, DegradationType.SUBSAMPLING, 0.8))
    #
    # baselines_df = baselines_df.append(read_baselines_for_degradation(baselines_dirs, DegradationType.SUBSTART, 0.2))
    # baselines_df = baselines_df.append(read_baselines_for_degradation(baselines_dirs, DegradationType.SUBSTART, 0.4))
    # baselines_df = baselines_df.append(read_baselines_for_degradation(baselines_dirs, DegradationType.SUBSTART, 0.6))
    # baselines_df = baselines_df.append(read_baselines_for_degradation(baselines_dirs, DegradationType.SUBSTART, 0.8))

    return baselines_df


def read_info_gain_results_all(in_dir):
    info_gain_df = read_finished_data(in_dir, DegradationType.NONE, None)

    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.ADD_NOISE, 10))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.ADD_NOISE, 100))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.ADD_NOISE, 200))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.ADD_NOISE, 300))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.ADD_NOISE, 400))

    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.SUBSAMPLING, 0.01))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.SUBSAMPLING, 0.05))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.SUBSAMPLING, 0.2))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.SUBSAMPLING, 0.4))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.SUBSAMPLING, 0.6))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.SUBSAMPLING, 0.8))

    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.SUBSTART, 0.01))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.SUBSTART, 0.05))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.SUBSTART, 0.2))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.SUBSTART, 0.4))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.SUBSTART, 0.6))
    info_gain_df = info_gain_df.append(read_finished_data(in_dir, DegradationType.SUBSTART, 0.8))

    info_gain_df = info_gain_df.sort_values(['user_id', 'traj_index'])  # This line is important for correctly calculating percent change

    return info_gain_df


def get_data_with_columns(
        df,
        trajectory_interval, query_pricing_type,
        degradation_type, degradation_value,
        transformation_type, start_prior, previous_purchases,
        grid_cell_len=1000, default_location_measurement_std=3,
        reconstruction_method=ReconstructionMethod.GAUSSIAN_PROCESS,
        min_traj_size=None, max_traj_size=None):

    data = df[df['trajectory_interval'] == trajectory_interval.name]
    data = data[data['query_pricing_type'] == query_pricing_type.name]
    data = data[data['degradation_type'] == degradation_type.name]
    if degradation_type == DegradationType.ADD_NOISE:
        data = data[np.isclose(data['added_noise_magnitude'], degradation_value)]
    elif degradation_type == DegradationType.SUBSAMPLING or \
            degradation_type == DegradationType.SUBSTART or \
            degradation_type == DegradationType.SUB_TIME:
        data = data[np.isclose(data['subsampling_ratio'], degradation_value)]

    data = data[data['transformation_type'] == transformation_type.name]
    data = data[np.isclose(data['eval_grid_cell_len'], grid_cell_len)]
    data = data[np.isclose(data['default_measurement_std'], default_location_measurement_std)]

    if 'start_prior' in data.columns:
        data = data[data['start_prior'] == start_prior.name]
    if 'previous_purchases' in data.columns:
        data = data[data['previous_purchases'] == previous_purchases.name]
    if 'previous_purchases' in data.columns:
        data = data[data['reconstruction_method'] == reconstruction_method.name]

    if min_traj_size is not None:
        data = data[data['traj_size'] >= min_traj_size]
    if max_traj_size is not None:
        data = data[data['traj_size'] <= max_traj_size]

    return data


def user_traj_to_tuple(x):
    return x['user_id'], x['traj_index']


def get_finished_user_trajectory_indexes(
        in_dir,
        trajectory_interval, query_pricing_type,
        degradation_type, degradation_value,
        transformation_type, start_prior, previous_purchases,
        grid_cell_len=1000, default_location_measurement_std=3,
        reconstruction_method=ReconstructionMethod.GAUSSIAN_PROCESS,
        min_traj_size=None, max_traj_size=None) -> Set[Tuple[int, int]]:
    """ Get (user_id, trajectory_index) of trajectories that we finished quantifying their VOI
    """

    df = read_finished_data(in_dir, degradation_type, degradation_value)
    if df is None:
        return set()

    df = get_data_with_columns(
        df,
        trajectory_interval=trajectory_interval,
        query_pricing_type=query_pricing_type,
        degradation_type=degradation_type,
        degradation_value=degradation_value,
        transformation_type=transformation_type,
        start_prior=start_prior,
        previous_purchases=previous_purchases,
        grid_cell_len=grid_cell_len,
        default_location_measurement_std=default_location_measurement_std,
        reconstruction_method=reconstruction_method,
        min_traj_size=min_traj_size, max_traj_size=max_traj_size)

    tmp = df.apply(user_traj_to_tuple, 1)
    return set(tmp)
