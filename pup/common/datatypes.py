"""
Common data types
"""

from typing import Dict, List

from pup.common.constants import DEFAULT_DATE_PATTERN


class Checkin(object):
    """ Object to store and manipulate a checkin

    Note about time: Datetime field is converted directly from raw time string.
    If there is no timestamp available in raw data, timestamp field is extracted from datetime field.
    If there is timestamp available in raw data, timestamp field is converted directly from raw data.
    Thus, timestamp field may not be the same with timestamp extracted from datetime field due to timezone difference

    Attributes
    ----------
    user_id: int, or str
        user id
    c_id: int
        id of check-in
    timestamp: int
        timestamp
    datetime: datetime
        datetime
    lat: float
        latitude
    lon: float
        longitude
    location_id: int, or str
        id of the check-in location
    x: float
        x in (x, y) coordinates
    y: float
        y in (x, y) coordinates
    trajectory_idx: int
        index of the trajectory containing this checkin if any
    """
    c_id = None
    user_id = None
    timestamp = None
    datetime = None
    lat = None
    lon = None
    measurement_std = None  # standard deviation of the location measurement
    location_id = None
    x = None  # converted to Oxy coordinate
    y = None  # converted to Oxy coordinate
    trajectory_idx = None

    def __init__(self, c_id, user_id, timestamp, datetime, lat, lon, measurement_std, location_id, trajectory_idx):
        """
        Initialize a checkin with given values from datasets

        Parameters
        ----------
        c_id: int, or str
            id of check-in
        user_id: int, or str
            user id
        timestamp: int
            timestamp
        datetime: datetime
            datetime
        lat: float
            latitude
        lon: float
            longitude
        location_id: int, or str
            id of the check-in location
        """
        self.c_id = c_id
        self.user_id = user_id
        self.timestamp = timestamp
        self.datetime = datetime
        self.lat = lat
        self.lon = lon
        self.measurement_std = measurement_std
        self.location_id = location_id
        self.trajectory_idx = trajectory_idx

    def __str__(self) -> str:
        return "Checkin(c_id={c_id}, user_id={user_id}, timestamp={timestamp}, datetime={datetime}, " \
               "lat={lat}, lon={lon}, measurement_std={measurement_std}, location_id={location_id}, x={x}, y={y})".format(**vars(self))

    def get_date_str(self, pattern: str = None) -> str:
        """ Get the string representation of the date of this checkin
        """
        if pattern is None:
            pattern = DEFAULT_DATE_PATTERN

        return self.datetime.strftime(pattern)


UserId = int
CheckinId = int

CheckinList = List[Checkin]  # list of checkins
Traj = CheckinList           # A trajectory is basically a sorted list of checkins
TrajList = List[Traj]        # A list of trajectories
TrajDataset = Dict[UserId, TrajList]  # A trajectory dataset is a mapping from users to their trajectories
