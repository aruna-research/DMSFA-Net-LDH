
import random
import numpy as np
import tensorflow as tf
from config import SEED

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from train import run_experiment

if __name__ == "__main__":
    run_experiment()

