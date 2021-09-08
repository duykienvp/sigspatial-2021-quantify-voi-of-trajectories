import math

from pup.common.constants import EARTH_RADIUS_METERS


def to_local_coordinates(orig_lat: float, orig_lon: float, lat: float, lon: float) -> tuple:
    """
    Convert (lat, lon) coordinates to coordinates in meters with (orig_lat, orig_lon) as the origin.

    Parameters
    ----------
    orig_lat: float
        latitude of origin
    orig_lon: float
        longitude of origin
    lat: float
        latitude of check-in
    lon: float
        longitude of check-in

    Returns
    -------
    tuple
        (vertical, horizontal) coordinates in meters (corresponding to latitude, longitude)
    """
    y_meters = cal_subtended_latitude_to_meters(lat - orig_lat)
    x_meters = cal_subtended_longitude_to_meters(lon - orig_lon, orig_lat)
    return y_meters, x_meters


def cal_subtended_latitude_to_meters(subtended_lat: float) -> float:
    """
    Calculate vertical coordinate in meters of a point at (subtended_lat, 0) with origin (0, 0).
    This is a rough estimate and should only be used when the `subtended_lat` is small enough.

    Parameters
    ----------
    subtended_lat: float
        coordinate in latitude dimension

    Returns
    -------
    float
        vertical coordinate in meters
    """
    return math.pi * EARTH_RADIUS_METERS * subtended_lat / 180.0


def cal_subtended_longitude_to_meters(subtended_lon: float, lat: float) -> float:
    """
    Calculate horizontal coordinate in meters of a point at (lat, subtended_lon) with origin at (lat, 0).
    This is a rough estimate and should only be used when the `subtended_lon` is small enough.

    Parameters
    ----------
    subtended_lon: float
        coordinate in longitude dimension
    lat: float
        latitude

    Returns
    -------
    float
        horizontal coordinate in meters
    """
    cos_lat = math.cos(math.pi * lat / 180.0)
    return math.pi * EARTH_RADIUS_METERS * subtended_lon * cos_lat / 180.0