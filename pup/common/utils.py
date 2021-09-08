import random
import numpy as np
import tensorflow as tf


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
