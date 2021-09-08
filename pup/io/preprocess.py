"""
Utility function for data pre-processing
"""
import logging
import random
from collections import defaultdict
from datetime import datetime

from pup.common import constants
from pup.common.coordinate_util import to_local_coordinates
from pup.common.datatypes import CheckinList, TrajList, Traj
from pup.common.enums import TrajectoryIntervalType

logger = logging.getLogger(__name__)


def checkins_to_location_coordinates(checkins: list, orig_lat: float, orig_lon: float):
    """
    Convert lat/lon coordinates of check-ins to y/x in meters. These y/x values are updated to check-ins themselves

    Parameters
    ----------
    checkins: list
        list of check-ins
    orig_lat: float
        latitude of origin
    orig_lon: float
        longitude of origin
    """
    for c in checkins:
        c.y, c.x = to_local_coordinates(orig_lat, orig_lon, c.lat, c.lon)


def split_to_trajectories(checkins: CheckinList, timezone, interval_type: TrajectoryIntervalType) -> TrajList:
    """ Separating checkins into trajectories by day in a certain time zone

    Parameters
    ----------
    checkins
        checkins list
    timezone
        timezone for separating trajectories

    Returns
    -------
    TrajDataset
        dict of user_id --> check-ins
    """
    traj_list = list()
    if not checkins:
        return traj_list

    sorted_checkins = sorted(checkins, key=lambda x: x.timestamp)

    prev_dt = datetime.fromtimestamp(sorted_checkins[0].timestamp, tz=timezone)
    prev_traj_idx = sorted_checkins[0].trajectory_idx
    current_traj = list()

    for c in sorted_checkins:
        dt = datetime.fromtimestamp(c.timestamp, tz=timezone)

        if interval_type == TrajectoryIntervalType.INTERVAL_GAP_THRESHOLD:
            # Gaps
            is_same_traj = (dt.timestamp() - prev_dt.timestamp()) <= constants.NUM_SECONDS_IN_5_MINUTE
        else:
            raise ValueError('Invalid interval type: {}'.format(interval_type))

        if not is_same_traj:
            # New trajectory
            traj_list.append(current_traj)
            current_traj = list()

        current_traj.append(c)
        prev_dt = dt
        prev_traj_idx = c.trajectory_idx

    if current_traj:
        traj_list.append(current_traj)
    else:
        logger.debug('Empty here end')

    return traj_list


def clean_checkins(checkins: CheckinList) -> CheckinList:
    """ Clean checkins by:
      - Removing duplicate checkins: 2 check-ins are duplicates if they are in the same timestamp
    """
    sorted_checkins = sorted(checkins, key=lambda x: x.timestamp)

    timestamps = set()
    cleaned_checkins = list()

    for c in sorted_checkins:
        if c.timestamp not in timestamps:
            cleaned_checkins.append(c)
            timestamps.add(c.timestamp)

    return cleaned_checkins


def assign_checkin_ids(checkins: CheckinList):
    """ Assign checkin id for each check in in a list, starting from 0

    :param checkins: list of checkins
    """
    for i in range(len(checkins)):
        checkins[i].c_id = i


def find_max_gap(trajectory: Traj) -> int:
    """ Find the maximum gap between 2 consecutive points in a trajectory

    :param trajectory: the trajectory
    :return: the maximum gap in seconds or 0 if len(trajectory) <= 1
    """
    max_gap = 0
    len_traj = len(trajectory)

    if len_traj <= 1:
        return max_gap

    for i in range(1, len_traj):
        gap = int(trajectory[i].timestamp - trajectory[i-1].timestamp)
        max_gap = max(max_gap, gap)

    return max_gap
