"""
Utility functions for data input/output
"""

import logging
import csv
import os
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd

import pytz

from pup.common import rectangle
from pup.common.constants import GEOLIFE_RAW_NUM_LINES_SKIPS, GEOLIFE_RAW_TIMEZONE, GEOLIFE_BEIJING_TIMEZONE
from pup.common.datatypes import TrajDataset, CheckinList, Checkin, TrajList, UserId
from pup.common.enums import DatasetType, AreaCode, TrajectoryIntervalType
from pup.common.rectangle import Rectangle
from pup.config import Config
from pup.io import preprocess

logger = logging.getLogger(__name__)


def load_data_by_config(at_subdir_idx=None, traj_interval=None) -> TrajDataset:
    """
    Load data from configuration.
    Also convert to local (x, y) coordinates

    Returns
    -------
    TrajDataset
        a trajectory dataset: mapping from user to their trajectories
    """
    data_dir = Config.data_dir
    dataset_type = Config.dataset_type
    area_code = Config.eval_area_code
    if traj_interval is None:
        traj_interval = Config.trajectory_interval

    rect = rectangle.get_rectangle_for_area(area_code)
    tz = GEOLIFE_BEIJING_TIMEZONE if area_code == AreaCode.BEIJING else None

    areas = [rect]

    data: TrajDataset = load_data(data_dir, dataset_type, areas, tz, traj_interval, at_subdir_idx)

    # convert_to_local_coordinates
    orig_x, orig_y = rect.get_mid_point()

    for user_id in data.keys():
        for traj in data[user_id]:
            preprocess.checkins_to_location_coordinates(traj, orig_y, orig_x)  # latitude = y, longitude = x

    logger.info('Loaded {} user data from {}'.format(len(data), data_dir))

    return data


def load_data(path: str,
              dataset_type: DatasetType,
              areas, tz,
              traj_interval: TrajectoryIntervalType,
              at_subdir_idx=None) -> Optional[TrajDataset]:
    """
    Load data from file

    Parameters
    ----------
    path: str
        path to dataset, either folder or file
    dataset_type: DatasetType
        type of dataset
    areas: list
        the data should belong to the intersection of these areas
    tz: str
        timezone for separating trajectories
    traj_interval
        trajectory interval

    Returns
    -------
    TrajDataset
        a trajectory dataset: mapping from user to their trajectories
    """
    if dataset_type == DatasetType.GEOLIFE_RAW:
        return load_dataset_geolife_raw(path, areas, tz, traj_interval, at_subdir_idx)
    else:
        logger.error("Invalid dataset type: {}".format(dataset_type))
        return None


def load_dataset_geolife_raw(path: str, areas, tz, trajectory_interval, at_subdir_idx=None) -> TrajDataset:
    """
    Load a GeoLife dataset in raw format.
    The path should point to a folder in which each sub-folder contains data of a user.
    In each user's folder, there is a list of files.
    Each of these files is data of that user with a separate starting time
    See: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/

    Parameters
    ----------
    path: str
        folder path
    areas: list
        the data should belong to the intersection of these areas
    tz: str
        timezone for separating trajectories
    trajectory_interval
        type of trajectory interval

    Returns
    -------
    TrajDataset
        a trajectory dataset: mapping from user to their trajectories
    """
    data: Dict[UserId, TrajList] = dict()

    start_checkin_id = 0
    timezone = pytz.timezone(tz) if tz is not None else pytz.timezone(GEOLIFE_RAW_TIMEZONE)
    # Each folder in this data path contains data of a user with the name of the folder as the user id
    sub_dirs = sorted(list_sub_dirs(path))
    for subdir_idx in range(len(sub_dirs)):
        if at_subdir_idx is not None and at_subdir_idx != subdir_idx:
            # if we asked to read data at a specific subdir idx and this is not it, then continue
            continue

        subdir = sub_dirs[subdir_idx]
        logger.info('Loading {}, {}/{}'.format(subdir, subdir_idx + 1, len(sub_dirs)))
        try:
            user_id = int(subdir)  # user id is the name of the folder containing data of that user
            user_path = os.path.join(path, subdir)
            user_path = os.path.join(user_path, 'Trajectory')
            user_file_names = sorted(list_files(user_path))
            checkins = list()
            user_original_traj_idx = 0
            for user_file_name in user_file_names:
                if user_file_name[0].isdigit():
                    more_checkins = read_checkin_file_geolife_raw(
                        os.path.join(user_path, user_file_name),
                        user_id,
                        start_checkin_id,
                        areas,
                        timezone
                    )

                    # Add the information about the original trajectory
                    for c in more_checkins:
                        c.trajectory_idx = user_original_traj_idx
                    user_original_traj_idx += 1

                    checkins.extend(more_checkins)
                    start_checkin_id += len(more_checkins)

            checkins = preprocess.clean_checkins(checkins)
            preprocess.assign_checkin_ids(checkins)

            # by default, we separate by day
            trajectories = preprocess.split_to_trajectories(checkins, timezone, trajectory_interval)
            data[user_id] = trajectories

            logger.info('user_id {}, {} checkins, {} trajectories'.format(user_id, len(checkins), len(data[user_id])))

            # Check
            # for traj in data[user_id]:
            #     assert len(traj) > 0, 'Empty trajectory'
            #
            #     for i in range(1, len(traj)):
            #         gap = traj[i].timestamp - traj[i-1].timestamp
            #         assert gap <= Config.trajectory_interval_gap_threshold, 'Big gap in user {}'.format(user_id)



            # assert len(checkins) == sum([len(t) for t in data[user_id]]), 'Not the same number of checkins'
            # for traj in data[user_id]:
            #     assert len(traj) > 0, 'Empty trajectory'
            #     dt = datetime.fromtimestamp(traj[0].timestamp, tz=timezone)
            #     for c in traj:
            #         dt2 = datetime.fromtimestamp(c.timestamp, tz=timezone)
            #         is_same = is_same_day(dt, dt2)
            #         if trajectory_interval == TrajectoryIntervalType.INTERVAL_HOUR:
            #             is_same = is_same_hour(dt, dt2)
            #         assert is_same, 'Not same hour within trajectory'

        except ValueError as err:
            logger.error('Error reading {}: {}'.format(path, err))

    return data


def read_checkin_file_geolife_raw(
        filepath: str, user_id: int, start_checkin_id: int = None, areas=None, target_tz=None) -> CheckinList:
    """ Read checkins from file for GeoLife.
    For format, see: https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/

    Parameters
    ----------
    filepath
        file path
    user_id
        user id
    start_checkin_id
        the starting check in id, if None, check-in id starts from 0
    areas: list
        the data should belong to the intersection of these areas
    target_tz:
        target timezone

    Returns
    -------
    list
        list of checkins in this file that belongs to the intersection of these areas
    """
    data = list()

    c_id = 0 if start_checkin_id is None else start_checkin_id

    measurement_std = Config.eval_default_location_measurement_std

    with open(filepath, 'r') as csv_file:
        for i in range(GEOLIFE_RAW_NUM_LINES_SKIPS):
            next(csv_file)

        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            lat = float(row[0])
            lon = float(row[1])

            # Get time
            date_str = str(row[5])
            time_str = str(row[6])
            datetime_str = date_str + ' ' + time_str
            t = datetime.fromisoformat(datetime_str)
            checkin_time = datetime(t.year, t.month, t.day, t.hour, t.minute, t.second,
                                    tzinfo=pytz.timezone(GEOLIFE_RAW_TIMEZONE))
            if target_tz is not None:
                checkin_time = datetime.fromtimestamp(checkin_time.timestamp(), tz=target_tz)

            c = Checkin(c_id=None,
                        user_id=user_id,
                        timestamp=int(checkin_time.timestamp()),
                        datetime=checkin_time,
                        lat=lat,
                        lon=lon,
                        measurement_std=measurement_std,
                        location_id=None,
                        trajectory_idx=None)

            # check if check-in is in these areas first
            if check_checkin_in_areas(c, areas):
                c.c_id = c_id
                c_id += 1

                data.append(c)

    return data


def check_checkin_in_areas(c: Checkin, areas=None) -> bool:
    """ Check whether a checkin is inside the intersection of these areas.
    If the areas list is None, return `True`

    Parameters
    ----------
    c: Checkin
        the checkin
    areas: list
        the data should belong to the intersection of these areas

    Returns
    -------
    bool
        whether a checkin is inside one of the intersection of these areas.
    """
    ok = True
    if areas is not None:
        for area in areas:
            ok = ok and check_checkin_in_rect(c, area)

    return ok


def check_checkin_in_rect(checkin: Checkin, rect: Rectangle) -> bool:
    """
    Check whether a checkin is inside a rectangle

    Parameters
    ----------
    checkin: Checkin
        the checkin
    rect: Rectangle
        rectangle

    Returns
    -------
    bool
        whether a checkin is inside the rectangle
    """
    return rect.contain(checkin.lon, checkin.lat)


def list_sub_dirs(path: str) -> List[str]:
    """ List sub-directories inside a directory

    Parameters
    ----------
    path
        directory path

    Returns
    -------
    list
        list of names of sub-directories
    """
    return [f.name for f in os.scandir(path) if f.is_dir()]


def list_files(path: str) -> List[str]:
    """ List files inside a directory

    Parameters
    ----------
    path
        directory path

    Returns
    -------
    list
        list of file names
    """
    return [f.name for f in os.scandir(path) if os.path.isfile(f)]


def write_line(file: str, value: str):
    """ Write a value line to file. A newline is added.
    """
    with open(file, 'a+') as f:
        f.write(value + '\n')