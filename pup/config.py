"""
Configuration handler: loading logging and program configuration
NOTE: This file must be included to be able to use the service correctly
"""

import logging
import logging.config
import os
import sys

import yaml

from pup.common.enums import DatasetType, AreaCode, DegradationType, \
    ReconstructionMethod, GpKernelType, FrameworkType, \
    TrajectoryIntervalType, PricingType, TransformationType, StartPriorType, \
    PreviousPurchaseType

CONFIG_FILE = 'config.yml'
LOG_CONFIG_FILE = 'logging.yml'


def parse_dashed_list(value: str):
    """

    Parameters
    ----------
    value: str
        string with 2 numbers separated by 1 dash

    Returns
    -------
    list
        list from those 2 numbers or `None` if error occurred

    Examples
    --------
    >>> parse_dashed_list('1-5')
    [1, 2, 3, 4, 5]
    """
    dash_pos = value.find('-')
    if dash_pos != -1:
        s = int(value[:dash_pos])
        t = int(value[dash_pos + 1:])
        return list(range(s, t + 1))
    return None


class Config(object):
    """ Configuration
    """
    is_config_loaded = False

    project_dir = None

    # data
    data_dir = None
    data_dir_name = None
    dataset_type = None

    output_dir = None
    output_file = None

    trajectory_interval = None

    # Query
    query_pricing_type = None
    query_transformation_type = None
    query_degradation_type = None
    query_random_seed = None
    query_subsampling_ratio = None
    query_add_noise_magnitude = None
    query_start_prior = None
    query_previous_purchases = None

    # Reconstruction
    reconstruction_method = None
    reconstruction_gp_framework = None
    reconstruction_gp_kernel = None
    reconstruction_max_training_size = None
    reconstruction_overlapping_size = None

    start_user_id = None
    start_traj_idx = None
    end_user_id = None

    # evaluation
    eval_area_code = None
    eval_grid_cell_len_x = None
    eval_grid_cell_len_y = None
    eval_grid_boundary_order = None
    eval_default_location_measurement_std = None

    eval_num_checkins_per_user = None


    # method specific
    find_actual_count = None
    fmc_budget_from_cost_percentage = None
    fmc_budget_from_probing = None
    budget = None
    start_price = -1
    start_std_ratio = None
    probing_parallel_find_best_buying_action = None
    probing_buy_singly = None
    probing_price_increment_factor = None
    probing_should_check_inout = None
    probing_should_check_only_one_next_price = None
    probing_probability_stopping = None
    probing_quantization_len = None
    probing_extended_cell_sigma_factor = None
    probing_point_inside_stop_threshold = None

    # privacy_distributions
    dist_privacy_should_random = None
    dist_user_privacy_level_loc = None
    dist_user_privacy_level_scale = None
    dist_user_privacy_level_random_seed = None

    dist_user_loc_sensitivity_loc = None
    dist_user_loc_sensitivity_scale = None
    dist_user_loc_sensitivity_random_seed = None

    price_from_noise_func_rate = None

    standard_deviation_from_noise_func_initial_value = None
    standard_deviation_from_noise_func_rate = None

    free_data_price_threshold = None

    @staticmethod
    def load_config(file, reload_config=False):
        """ Load config from a file.
        Args:
            file: config file in YAML format
            reload_config: should we reload config
        """
        logger = logging.getLogger(__name__)
        # Load the configuration file
        if file is None:
            logger.error('No config file provided')
            return

        if not Config.is_config_loaded or reload_config:
            with open(file, 'r') as conf_file:
                cfg = yaml.safe_load(conf_file)
                # print(cfg)

                # Load config to variables

                if 'project_dir' in cfg:
                    Config.project_dir = cfg['project_dir']

                # data files
                if 'data' in cfg:
                    data_cfg = cfg['data']

                    if 'data_dir' in data_cfg:
                        Config.data_dir = os.path.join(Config.project_dir, data_cfg['data_dir'])

                        Config.data_dir_name = os.path.basename(os.path.normpath(Config.data_dir))

                    if 'dataset_type' in data_cfg:
                        Config.dataset_type = DatasetType[data_cfg['dataset_type'].upper()]

                # output
                if 'output' in cfg:
                    output_cfg = cfg['output']

                    if 'output_dir' in output_cfg:
                        Config.output_dir = os.path.join(Config.project_dir, output_cfg['output_dir'])
                    if 'output_file' in output_cfg:
                        Config.output_file = os.path.join(Config.output_dir, output_cfg['output_file'])

                # trajectory interval
                if 'trajectory_interval' in cfg:
                    Config.trajectory_interval = TrajectoryIntervalType[cfg['trajectory_interval'].upper()]

                # degradation
                if 'query' in cfg:
                    query_cfg = cfg['query']

                    if 'pricing_type' in query_cfg:
                        Config.query_pricing_type = PricingType[query_cfg['pricing_type'].upper()]

                    if 'transformation_type' in query_cfg:
                        Config.query_transformation_type = TransformationType[query_cfg['transformation_type'].upper()]

                    if 'degradation_type' in query_cfg:
                        Config.query_degradation_type = DegradationType[query_cfg['degradation_type'].upper()]

                    if 'random_seed' in query_cfg:
                        Config.query_random_seed = int(query_cfg['random_seed'])

                    if 'subsampling_ratio' in query_cfg:
                        Config.query_subsampling_ratio = float(query_cfg['subsampling_ratio'])

                    if 'add_noise_magnitude' in query_cfg:
                        Config.query_add_noise_magnitude = float(query_cfg['add_noise_magnitude'])

                    if 'start_prior' in query_cfg:
                        Config.query_start_prior = StartPriorType[query_cfg['start_prior'].upper()]

                    if 'previous_purchases' in query_cfg:
                        Config.query_previous_purchases = PreviousPurchaseType[query_cfg['previous_purchases'].upper()]

                # Reconstruction
                if 'reconstruction' in cfg:
                    reconstruction_cfg = cfg['reconstruction']
                    if 'method' in reconstruction_cfg:
                        Config.reconstruction_method = ReconstructionMethod[reconstruction_cfg['method'].upper()]
                    if 'framework' in reconstruction_cfg:
                        Config.reconstruction_gp_framework = FrameworkType[reconstruction_cfg['framework'].upper()]
                    if 'gp_kernel' in reconstruction_cfg:
                        Config.reconstruction_gp_kernel = GpKernelType[reconstruction_cfg['gp_kernel'].upper()]
                    if 'max_training_size' in reconstruction_cfg:
                        Config.reconstruction_max_training_size = int(reconstruction_cfg['max_training_size'])
                    if 'overlapping_size' in reconstruction_cfg:
                        Config.reconstruction_overlapping_size = int(reconstruction_cfg['overlapping_size'])

                if 'start_user_id' in cfg:
                    Config.start_user_id = int(cfg['start_user_id'])

                if 'start_traj_idx' in cfg:
                    Config.start_traj_idx = int(cfg['start_traj_idx'])

                if 'end_user_id' in cfg:
                    Config.end_user_id = int(cfg['end_user_id'])


                # evaluation
                if 'evaluation' in cfg:
                    evaluation_cfg = cfg['evaluation']

                    if 'area_code' in evaluation_cfg:
                        Config.eval_area_code = AreaCode[evaluation_cfg['area_code'].upper()]

                    if 'grid_cell_len_x' in evaluation_cfg:
                        Config.eval_grid_cell_len_x = evaluation_cfg['grid_cell_len_x']

                    if 'grid_cell_len_y' in evaluation_cfg:
                        Config.eval_grid_cell_len_y = evaluation_cfg['grid_cell_len_y']

                    if 'grid_boundary_order' in evaluation_cfg:
                        Config.eval_grid_boundary_order = int(evaluation_cfg['grid_boundary_order'])

                    if 'default_location_measurement_std' in evaluation_cfg:
                        Config.eval_default_location_measurement_std = float(
                            evaluation_cfg['default_location_measurement_std'])

    @staticmethod
    def get_config_str() -> str:
        """
        Get string representation of all configuration

        Returns
        -------
        str
            string representation of all the configurations
        """
        values = []
        for k, v in Config.__dict__.items():
            tmp = str(k) + '=' + str(v) + '\n'
            values.append(tmp)
        values.sort()
        res = ''.join(values)
        return res


def setup_logging(default_path='logging.yml', default_level=logging.INFO, env_key='LOG_CFG'):
    """ Setup logging configuration

    Parameters
    ----------
    default_path: str
        default logging file path
    default_level:
        default logging level
    env_key: str
        environment key to get logging config
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt', encoding=sys.getfilesystemencoding()) as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logging_config_file = os.path.join(dir_path, LOG_CONFIG_FILE)
setup_logging(default_path=logging_config_file)


Config.load_config(os.path.join(dir_path, CONFIG_FILE))
