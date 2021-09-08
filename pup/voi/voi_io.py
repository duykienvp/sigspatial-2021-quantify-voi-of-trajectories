import errno
import logging
import os
import pickle

import gpflow

from pup.common.enums import DegradationType, TransformationType, PreviousPurchaseType
from pup.config import Config
from pup.voi import voi_utils

logger = logging.getLogger(__name__)

MODEL_PRED_FILE_NAME_EXTENSION = '.pkl'

QUANTIFYING_VOI_MODELS_PREFIX = 'scp_models'
QUANTIFYING_VOI_PREDICTION_PREFIX = 'scp_pred'


# OUTPUT DIR


def get_reconstruction_models_and_pred_dir(prefix, should_create=True, start_dir=None, previous_purchases=None) -> str:
    """ Create the output directory containing reconstruction output

    :param prefix: prefix
    :param should_create: should we create the directory or not
    :param start_dir: the starting directory. If None, output_dir in configuration is used
    :return: the dir path
    """
    if start_dir is None:
        start_dir = Config.output_dir
    out_dir = os.path.join(start_dir, prefix)

    out_dir = os.path.join(out_dir, '{}'.format(Config.trajectory_interval.name))

    # We train/predict all with 'PreviousPurchaseType.NONE' if not provided
    if previous_purchases is None:
        previous_purchases = Config.query_previous_purchases
        if previous_purchases is None:
            previous_purchases = PreviousPurchaseType.NONE
    out_dir = os.path.join(
        out_dir,
        'start_prior_{}_prev_purchases_{}'.format(
            Config.query_start_prior.name,
            previous_purchases.name)
    )

    out_dir = os.path.join(
        out_dir,
        'train_size_{}_overlapping_size_{}'.format(
            Config.reconstruction_max_training_size,
            Config.reconstruction_overlapping_size)
    )

    out_dir = os.path.join(out_dir, 'reconstruct_{}'.format(Config.reconstruction_method.name))
    out_dir = os.path.join(out_dir, 'framework_{}'.format(Config.reconstruction_gp_framework.name))

    if prefix == QUANTIFYING_VOI_PREDICTION_PREFIX:
        out_dir = os.path.join(out_dir, 'pricing_{}'.format(Config.query_pricing_type.name))

    if should_create:
        try:
            os.makedirs(out_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    return out_dir


# OUTPUT FILE NAME

def get_single_component_pricing_reconstruction_models_and_pred_filename(
        user_id, traj_idx,
        prefix,
        degradation_type=None, degradation_value=None, previous_purchases=None,
        is_prediction=False, extension=MODEL_PRED_FILE_NAME_EXTENSION) -> str:
    """
    Get the file of saved models file for Single component pricing

    :return: file object
    """
    if degradation_type is None:
        degradation_type, degradation_value = voi_utils.get_degradation_from_config()

    start_prior = Config.query_start_prior

    # We train/predict all with 'PreviousPurchaseType.NONE' if not provided
    if previous_purchases is None:
        previous_purchases = Config.query_previous_purchases
        if previous_purchases is None:
            previous_purchases = PreviousPurchaseType.NONE

    filename = '{}_degrade_{}_seed_{}_prior_{}_purchases_{}'.format(
        prefix,
        degradation_type.name,
        Config.query_random_seed,
        start_prior.name,
        previous_purchases.name
    )

    if degradation_type == DegradationType.SUBSAMPLING or \
            degradation_type == DegradationType.SUBSTART or \
            degradation_type == DegradationType.SUB_TIME:
        filename = '{}_sub_ratio_{:.3f}'.format(filename, degradation_value)
    elif degradation_type == DegradationType.ADD_NOISE:
        filename = '{}_noise_{}'.format(filename, int(degradation_value))
    elif degradation_type == DegradationType.NONE:
        filename = '{}_nodegrade'.format(filename)

    filename = '{}_notrans'.format(filename)

    if is_prediction:
        filename = '{}_{}'.format(filename, Config.query_pricing_type.name)

    filename = '{}_user_{}_traj_{}'.format(filename, user_id, traj_idx)
    filename += extension

    return filename


def get_dir_name_for_degradation(degradation_type, degradation_value) -> str:
    """
    Get the directory name for a degradation.
    For example: SUBSAMPLING with ratio 0.2 will give `sub02`

    :param degradation_type: type of degradation
    :param degradation_value: value of degradation
    :return: directory name
    :raise: `ValueError` if input is invalid
    """
    if degradation_type is None:
        degradation_type = Config.query_degradation_type

    if degradation_type is None:
        raise ValueError('Invalid Degradation Type: {}'.format(degradation_type))

    degradation_dir = 'unknown'

    if degradation_type == DegradationType.NONE:
        degradation_dir = 'nodegrade'

    elif degradation_type == DegradationType.SUBSAMPLING:
        if degradation_value is None:
            degradation_value = Config.query_subsampling_ratio

        degradation_dir = 'sub0{}'.format(int(degradation_value * 10))
        if degradation_value < 0.1:
            degradation_dir = 'sub00{}'.format(int(degradation_value * 100))

    elif degradation_type == DegradationType.SUBSTART:
        if degradation_value is None:
            degradation_value = Config.query_subsampling_ratio

        degradation_dir = 'substart0{}'.format(int(degradation_value * 10))
        if degradation_value < 0.1:
            degradation_dir = 'substart00{}'.format(int(degradation_value * 100))

    elif degradation_type == DegradationType.SUB_TIME:
        if degradation_value is None:
            degradation_value = Config.query_subsampling_ratio

        degradation_dir = 'subtime0{}'.format(int(degradation_value * 10))

    elif degradation_type == DegradationType.ADD_NOISE:
        if degradation_value is None:
            degradation_value = Config.query_add_noise_magnitude

        degradation_dir = 'noise{}'.format(int(degradation_value))

    return degradation_dir


def get_out_dir_and_filename(
        user_id, traj_idx,
        start_dir, prefix, should_create,
        degradation_type, degradation_value, previous_purchases,
        is_prediction):

    degradation_dir = get_dir_name_for_degradation(degradation_type, degradation_value)

    start_dir = os.path.join(start_dir, degradation_dir)

    out_dir = get_reconstruction_models_and_pred_dir(
        prefix=prefix,
        should_create=should_create,
        start_dir=start_dir,
        previous_purchases=previous_purchases
    )
    filename = get_single_component_pricing_reconstruction_models_and_pred_filename(
        user_id=user_id, traj_idx=traj_idx,
        prefix=prefix,
        degradation_type=degradation_type, degradation_value=degradation_value, previous_purchases=previous_purchases,
        is_prediction=is_prediction)

    return out_dir, filename


# MODELS

def single_component_pricing_save_models(
        user_id, traj_idx, models, degradation_type=None, degradation_value=None, previous_purchases=None):
    """
    Save models

    :param user_id: user id
    :param traj_idx: trajectory index
    :param models: list of models to save
    """
    out_dir, filename = get_out_dir_and_filename(
        user_id=user_id, traj_idx=traj_idx,
        start_dir=Config.output_dir, prefix=QUANTIFYING_VOI_MODELS_PREFIX, should_create=True,
        degradation_type=degradation_type, degradation_value=degradation_value, previous_purchases=previous_purchases,
        is_prediction=False)

    # Extract and save model values to a single file
    models_to_save = list()
    for i in range(len(models)):
        gp_0, gp_1, scaler_x, scaler_y = models[i]
        values_0 = gpflow.utilities.read_values(gp_0)
        values_1 = gpflow.utilities.read_values(gp_1)

        models_to_save.append((values_0, values_1, scaler_x, scaler_y))

    out_file = os.path.join(out_dir, filename)

    with open(out_file, 'wb') as f:
        pickle.dump(models_to_save, f)

    logger.info('Saved model to: {}'.format(out_file))


def single_component_pricing_check_models_existed(
        user_id, traj_idx, degradation_type=None, degradation_value=None, previous_purchases=None):
    """
    Load models

    :param user_id: user id
    :param traj_idx: trajectory index
    :param previous_purchases: type of previous purchases or None if using it from Config
    :return: list of models to save or None if error occurred
    """
    out_dir, filename = get_out_dir_and_filename(
        user_id=user_id, traj_idx=traj_idx,
        start_dir=Config.output_dir, prefix=QUANTIFYING_VOI_MODELS_PREFIX, should_create=False,
        degradation_type=degradation_type, degradation_value=degradation_value, previous_purchases=previous_purchases,
        is_prediction=False)

    out_file_new_format = os.path.join(out_dir, filename)

    return os.path.isfile(out_file_new_format)


def single_component_pricing_load_models(user_id, traj_idx, degradation_type=None, degradation_value=None, previous_purchases=None):
    """
    Load models

    :param user_id: user id
    :param traj_idx: trajectory index
    :param framework_type: type of framework
    :return: list of models to save or None if error occurred
    """
    out_dir, filename = get_out_dir_and_filename(
        user_id=user_id, traj_idx=traj_idx,
        start_dir=Config.output_dir, prefix=QUANTIFYING_VOI_MODELS_PREFIX, should_create=False,
        degradation_type=degradation_type, degradation_value=degradation_value, previous_purchases=previous_purchases,
        is_prediction=False)

    # new format
    in_file = os.path.join(out_dir, filename)

    if os.path.isfile(in_file):
        with open(in_file, 'rb') as f:
            logger.info('Load model from: {}'.format(in_file))
            return pickle.load(f)

    return None

# PREDICTIONS


def single_component_pricing_save_predictions(user_id, traj_idx, sigmas_preds, degradation_type, degradation_value, previous_purchases):
    """
    Save models

    :param user_id: user id
    :param traj_idx: trajectory index
    :param sigmas_preds: list of sigmas preds
    """
    out_dir, filename = get_out_dir_and_filename(
        user_id=user_id, traj_idx=traj_idx,
        start_dir=Config.output_dir, prefix=QUANTIFYING_VOI_PREDICTION_PREFIX, should_create=True,
        degradation_type=degradation_type, degradation_value=degradation_value, previous_purchases=previous_purchases,
        is_prediction=True)

    out_file = os.path.join(out_dir, filename)

    with open(out_file, 'wb') as f:
        pickle.dump(sigmas_preds, f)

    logger.info('Saved predictions to: {}'.format(out_file))


def quantify_voi_has_predictions(user_id, traj_idx, degradation_type=None, degradation_value=None, previous_purchases=None):
    """
    Save models

    :param user_id: user id
    :param traj_idx: trajectory index
    :returns: has predictions or not
    """
    out_dir, filename = get_out_dir_and_filename(
        user_id=user_id, traj_idx=traj_idx,
        start_dir=Config.output_dir, prefix=QUANTIFYING_VOI_PREDICTION_PREFIX, should_create=False,
        degradation_type=degradation_type, degradation_value=degradation_value, previous_purchases=previous_purchases,
        is_prediction=True)

    return os.path.isfile(os.path.join(out_dir, filename))


def voi_load_predictions(user_id, traj_idx, degradation_type=None, degradation_value=None, previous_purchases=None):
    """
    Save models

    :param user_id: user id
    :param traj_idx: trajectory index
    :returns:
        - y_preds - list of y_preds <br>
        - sigmas_preds - list of sigmas preds
    """
    out_dir, filename = get_out_dir_and_filename(
        user_id=user_id, traj_idx=traj_idx,
        start_dir=Config.output_dir, prefix=QUANTIFYING_VOI_PREDICTION_PREFIX, should_create=False,
        degradation_type=degradation_type, degradation_value=degradation_value, previous_purchases=previous_purchases,
        is_prediction=True)

    in_file = os.path.join(out_dir, filename)

    try:
        with open(in_file, 'rb') as f:
            logger.info('Load prediction from: {}'.format(in_file))
            return pickle.load(f)

    except OSError:
        logger.info('Predictions file not found for user {}, traj {}'.format(user_id, traj_idx))
        return None






