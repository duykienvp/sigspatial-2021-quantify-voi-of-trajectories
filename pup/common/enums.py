from enum import Enum, IntEnum


class DatasetType(Enum):
    """
    Types of datasets
    """
    GEOLIFE_RAW = 'GEOLIFE_RAW'


class AreaCode(Enum):
    """
    Area code for areas we experimented
    """
    LOS_ANGELES = 'LOS_ANGELES'
    BEIJING = 'BEIJING'
    GLOBAL = 'GLOBAL'


class TrajectoryIntervalType(Enum):
    """
    Types of trajectory intervals
    """
    INTERVAL_GAP_THRESHOLD = 'INTERVAL_GAP_THRESHOLD'  # DEfault gap is 5 minutes


class PricingType(Enum):
    """
    Type of pricing
    """
    BASELINES = 'BASELINES'  # All baselines : SIZE, DURATION, DISTANCE, ENTROPY
    HISTOGRAM_ENTROPY = 'HISTOGRAM_ENTROPY'  # Based on entropy of its histogram
    TRAVEL_DISTANCE = 'TRAVEL_DISTANCE'
    MARKOV_CHAIN_ENTROPY = 'MARKOV_CHAIN_ENTROPY'
    RECONSTRUCTION = 'RECONSTRUCTION'
    IG_TRAJ_DAY = 'IG_TRAJ_DAY'
    IG_TRAJ_DURATION = 'IG_TRAJ_DURATION'


class DegradationType(Enum):
    """
    Type of degradation
    """
    NONE = 'NONE'   # No transformation
    SUBSAMPLING = 'SUBSAMPLING'  # Subsample uniformly
    ADD_NOISE = 'ADD_NOISE'    # Add random Gaussian noises to each data point
    SUBSTART = 'SUBSTART'    # Cut a subsample from start to x%
    SUB_TIME = 'SUB_TIME'  # Uniformly randomly select x% of timestamps


class TransformationType(Enum):
    """
    Type of transformation
    """
    NONE = 'NONE'   # No transformation
    HISTOGRAM = 'HISTOGRAM'


class ReconstructionMethod(Enum):
    """
    Types of reconstruction methods
    """
    GAUSSIAN_PROCESS = 'GAUSSIAN_PROCESS'


class FrameworkType(Enum):
    GPFLOW = 'gpflow'


class GpKernelType(Enum):
    MATERN32 = 'MATERN32'
    MATERN12 = 'MATERN12'
    MATERN52 = 'MATERN52'
    RATIONALQUADRATIC = 'RATIONALQUADRATIC'
    SQUAREDEXPONENTIAL = 'SQUAREDEXPONENTIAL'


class StartPriorType(Enum):
    """
    Types of starting priors
    """
    UNIFORM_GRID = 'UNIFORM_GRID'  # Uniform distributions other the grid
    CENTERED_NORMAL = 'CENTERED_NORMAL' # Multivariate normal distribution centered within the grid


class PreviousPurchaseType(Enum):
    """
    Types of how we use previous purchases
    """
    NONE = 'NONE'  # No previous purchases
    FIRST_TRAJ = 'FIRST_TRAJ'  # Assume we bought the 1st trajectory of this seller before
    SAME_TRAJ_NOISE_300 = 'SAME_TRAJ_NOISE_300'  # Assume we bought the same trajectory but at worse noise (300m) and we REPLACE it with new data
    SAME_TRAJ_NOISE_400 = 'SAME_TRAJ_NOISE_400'  # Assume we bought the same trajectory but at worse noise (400m) and we REPLACE it with new data
    SAME_TRAJ_NOISE_300_COMBINED = 'SAME_TRAJ_NOISE_300_COMBINED'  # Assume we bought the same trajectory but at worse noise (300m) and we COMBINE it with new data
    SAME_TRAJ_NOISE_400_COMBINED = 'SAME_TRAJ_NOISE_400_COMBINED'  # Assume we bought the same trajectory but at worse noise (400m) and we COMBINE it with new data
    SAME_TRAJ_NOISE_300_COMBINED_RETRAINED = 'SAME_TRAJ_NOISE_300_COMBINED_RETRAINED'  # Assume we bought the same trajectory but at worse noise (300m) and we COMBINE it with new data and then retrain model with new data
    SAME_TRAJ_NOISE_400_COMBINED_RETRAINED = 'SAME_TRAJ_NOISE_400_COMBINED_RETRAINED'  # Assume we bought the same trajectory but at worse noise (400m) and we COMBINE it with new data and then retrain model with new data
    SAME_TRAJ_SUB_001 = 'SAME_TRAJ_SUB_001'
    SAME_TRAJ_SUB_005 = 'SAME_TRAJ_SUB_005'
    SAME_TRAJ_SUB_02 = 'SAME_TRAJ_SUB_02'
    SAME_TRAJ_SUB_001_RETRAINED = 'SAME_TRAJ_SUB_001_RETRAINED'
    SAME_TRAJ_SUB_005_RETRAINED = 'SAME_TRAJ_SUB_005_RETRAINED'
    SAME_TRAJ_SUB_02_RETRAINED = 'SAME_TRAJ_SUB_02_RETRAINED'
    SAME_TRAJ_SUB_START_001 = 'SAME_TRAJ_SUB_START_001'
    SAME_TRAJ_SUB_START_005 = 'SAME_TRAJ_SUB_START_005'
    SAME_TRAJ_SUB_START_02 = 'SAME_TRAJ_SUB_START_02'
    SAME_TRAJ_SUB_START_001_RETRAINED = 'SAME_TRAJ_SUB_START_001_RETRAINED'
    SAME_TRAJ_SUB_START_005_RETRAINED = 'SAME_TRAJ_SUB_START_005_RETRAINED'
    SAME_TRAJ_SUB_START_02_RETRAINED = 'SAME_TRAJ_SUB_START_02_RETRAINED'
    SAME_TRAJ_SUB_TIME_001 = 'SAME_TRAJ_SUB_TIME_001'
    SAME_TRAJ_SUB_TIME_02 = 'SAME_TRAJ_SUB_TIME_02'
