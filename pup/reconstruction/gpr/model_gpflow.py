"""
Gaussian Process using GPFlow library
"""
import datetime
import logging

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import set_trainable

from pup.common.enums import GpKernelType

logger = logging.getLogger(__name__)


def set_up_gpu():
    """
    Set up GPU such as using GPU or not, which GPU to use, how much RAM...
    """
    # User only 1 GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            gpu_idx = None
            # gpu_idx = 0

            if gpu_idx is None:
                tf.config.set_visible_devices([], 'GPU')
            else:
                tf.config.set_visible_devices(gpus[gpu_idx], 'GPU')
                memory_limit = 1200
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[gpu_idx],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

    #         # tf.config.set_visible_devices([], 'GPU')
    #         tf.config.set_visible_devices(gpus[0], 'GPU')
    # #        tf.config.set_visible_devices(gpus[1], 'GPU')

        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)

        # # Set dynamic memory allocation
        # for gpu_instance in gpus:
        #     tf.config.experimental.set_memory_growth(gpu_instance, True)


def construct_kernel_gpflow(kernel_type, measurement_std, kernel_std=1.0, kernel_variance_trainable=False):
    """ Construct a GPFlow kernel
    :return: a kernel
    """

    #print(np.square(measurement_std))
    if kernel_type == GpKernelType.MATERN32:
        k = gpflow.kernels.Matern32(lengthscales=1, variance=np.square(kernel_std))  # default kernel
    elif kernel_type == GpKernelType.MATERN12:
        k = gpflow.kernels.Matern12(lengthscales=1, variance=np.square(kernel_std))
    elif kernel_type == GpKernelType.MATERN52:
        k = gpflow.kernels.Matern52(lengthscales=1, variance=np.square(kernel_std))
    elif kernel_type == GpKernelType.RATIONALQUADRATIC:
        k = gpflow.kernels.RationalQuadratic(lengthscales=1, variance=np.square(kernel_std))
    elif kernel_type == GpKernelType.SQUAREDEXPONENTIAL:
        k = gpflow.kernels.SquaredExponential(lengthscales=1, variance=np.square(kernel_std))
    else:
        raise ValueError('Unknown kernel type: {}'.format(kernel_type))

    # THIS LINE IS FOR User 157, Traj 12, ADD_NOISE 100, PreviousPurchase NONE because of not invertible matrix
    # measurement_std += 0.01

    k += gpflow.kernels.White(variance=np.square(measurement_std))  # default kernel

    k.kernels[0].lengthscales.prior = tfp.distributions.Uniform(low=np.float64(0.01), high=np.float64(10))
    set_trainable(k.kernels[0].variance, kernel_variance_trainable)
    set_trainable(k.kernels[1].variance, False)

    return k


def optimize_model_gpr(X, y, kernel):
    """
    Optimize (i.e., actual training) model

    :param X:
    :param y:
    :param kernel:
    :return: trained model
    """
    # Prepare mean function
    if 1 < len(y):
        x_arr = np.squeeze(np.array(X))
        y_arr = np.squeeze(np.array(y))
        m, b = np.polyfit(x_arr, y_arr, 1)
        meanf = gpflow.mean_functions.Linear(A=m, b=b)
    else:
        meanf = None

    model = gpflow.models.GPR(data=(X, y), kernel=kernel, mean_function=meanf)
    model.likelihood.variance.assign(1e-5)

    gpflow.optimizers.Scipy().minimize(
        model.training_loss,
        variables=model.trainable_variables,
        options=dict(disp=False, maxiter=100)
    )

    return model


def construct_model_for_new_data(X, y, params, kernel_type, measurement_std):
    """
    Construct GPR model for new data

    :param X: x values
    :param y: y values
    :param params: some parameters need to be set. This comes from previously trained model that we want to keep
    :param kernel_type:
    :param measurement_std:
    :return:
    """
    kernel = construct_kernel_gpflow(kernel_type, 1, 1, False)
    if '.mean_function.A' in params:
        meanf = gpflow.mean_functions.Linear(A=params['.mean_function.A'][0][0], b=params['.mean_function.b'])
    else:
        meanf = None

    model = gpflow.models.GPR(data=(X, y), kernel=kernel, mean_function=meanf)

    if float(params['.likelihood.variance']) < 5e-6:
        params['.likelihood.variance'] = np.array(5e-6)

    gpflow.utilities.multiple_assign(model, params)
    model.kernel.kernels[1].variance = np.square(measurement_std)

    return model


def construct_fit_model_gpflow(X, y, kernel_type, measurement_std, kernel_stds, kernel_variance_trainable):
    """ Construct GPFlow Gaussian Process models for latitude and longitude separately and fit them

    :param X: feature matrix
    :param y: label matrix
    :param kernel_type: type of kernel
    :param measurement_std: default standard deviation of location measurement
    :return: tuple of trained models as (gp_lat, gp_lon)
    """
    gp_0 = construct_fit_model_gpflow_single(X, y[:, 0].reshape((-1, 1)), None, kernel_type, measurement_std[0], kernel_stds[0], kernel_variance_trainable)
    gp_1 = construct_fit_model_gpflow_single(X, y[:, 1].reshape((-1, 1)), None, kernel_type, measurement_std[1], kernel_stds[1], kernel_variance_trainable)

    return gp_0, gp_1


def construct_fit_model_gpflow_single(X, y, kernel, kernel_type, measurement_std, kernel_std, kernel_variance_trainable):
    """ Construct a GPFlow model and optimize it

    :param measurement_std:
    :param kernel_type:
    :param X: feature matrix
    :param y: label matrix
    :param kernel: provided kernel. If None, new kernel is constructed
    :return: a GPFlow model
    """
    s = datetime.datetime.now().timestamp()

    k = construct_kernel_gpflow(kernel_type, measurement_std, kernel_std, kernel_variance_trainable)

    model = optimize_model_gpr(X, y, k)

    # print_summary(model)

    time_train = datetime.datetime.now().timestamp() - s

    logger.info("gpflow time_train = {}".format(time_train))

    return model


def predict_gpflow(model, X_pred):
    """
    Make prediction for given input values

    :param model: model for prediction
    :param X_pred: scaled x values we need to predict for
    :return: y_pred, sigmas_pred
    """
    gp_lat, gp_lon = model

    y_pred_lat_tf, var_pred_lat_tf = gp_lat.predict_y(X_pred)
    y_pred_lon_tf, var_pred_lon_tf = gp_lon.predict_y(X_pred)

    y_pred_lat = y_pred_lat_tf.numpy()
    var_pred_lat = var_pred_lat_tf.numpy()
    y_pred_lon = y_pred_lon_tf.numpy()
    var_pred_lon = var_pred_lon_tf.numpy()

    sigmas_pred_lat = np.sqrt(var_pred_lat)
    sigmas_pred_lon = np.sqrt(var_pred_lon)

    # stack them into n x 2 matrix
    y_pred = np.hstack((y_pred_lat.reshape(-1, 1), y_pred_lon.reshape(-1, 1)))
    sigmas_pred = np.hstack((sigmas_pred_lat.reshape(-1, 1), sigmas_pred_lon.reshape(-1, 1)))

    return y_pred, sigmas_pred





