"""
Some constants used throughout the project
"""
NUM_SECONDS_IN_MINUTE = 60
NUM_SECONDS_IN_5_MINUTE = NUM_SECONDS_IN_MINUTE * 5
NUM_SECONDS_IN_HOUR = NUM_SECONDS_IN_MINUTE * 60
NUM_SECONDS_IN_2_HOURS = NUM_SECONDS_IN_HOUR * 2
NUM_SECONDS_IN_3_HOURS = NUM_SECONDS_IN_HOUR * 3
NUM_SECONDS_IN_4_HOURS = NUM_SECONDS_IN_HOUR * 4
NUM_SECONDS_IN_DAY = NUM_SECONDS_IN_HOUR * 24

FLOAT_IS_CLOSE_TOLERANCE = 1e-12   # the tolerance for float equality comparison

EARTH_RADIUS_METERS = 6371.0087714150598 * 1000  # Earth radius in meters

MIN_LAT = -90   # min value of latitude
MAX_LAT = 90    # min value of longitude
MIN_LON = -180  # max value of latitude
MAX_LON = 190   # max value of longitude

LOS_ANGELES_MAX_LAT = 34.342324    # min value of latitude of Los Angeles area
LOS_ANGELES_MIN_LAT = 33.699675    # min value of longitude of Los Angeles area
LOS_ANGELES_MAX_LON = -118.144458  # max value of latitude of Los Angeles area
LOS_ANGELES_MIN_LON = -118.684687  # max value of longitude of Los Angeles area

BEIJING_MAX_LAT = 40.06    # min value of latitude of Beijing area
BEIJING_MIN_LAT = 39.80    # min value of longitude of Beijing area
BEIJING_MAX_LON = 116.55  # max value of latitude of Beijing area
BEIJING_MIN_LON = 116.20  # max value of longitude of Beijing area

DEFAULT_DATE_PATTERN = '%Y-%m-%d'  # date pattern used for printing dates
DEFAULT_DATETIME_PATTERN = '%Y-%m-%d %H:%M:%S'  # date pattern used for printing dates

GEOLIFE_RAW_TIME_PATTERN = '%Y-%m-%d %H:%M:%S'  # time pattern in GeoLife dataset, used for parsing data
GEOLIFE_RAW_TIMEZONE = 'GMT'                    # timezone in GeoLife dataset, used for parsing data
GEOLIFE_BEIJING_TIMEZONE = 'Asia/Shanghai'      # timezone in experiment with GeoLife dataset
GEOLIFE_RAW_NUM_LINES_SKIPS = 6                 # number of lines in in GeoLife data file we should ignore
GOWALLA_TIME_PATTERN = '%Y-%m-%dT%H:%M:%S%z'  # time pattern in Gowalla dataset, used for parsing data
GOWALLA_LA_TIMEZONE = 'America/Los_Angeles'      # timezone in experiment with GeoLife dataset

SQRT_2 = 1.4142135623730951
