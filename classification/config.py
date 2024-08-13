import random
import os

import tensorflow as tf

### ===================================================== Configurations of Classification =====================================================================
random.seed(42)
tf.random.set_seed(42)

class LABEL():
    MINE = 1
    NOT_MINE = -1
    NOT_SURE = 0

LABELED_DATASET_PATH = "/mnt/mining/labeled_mine_dataset"
INCLUDE_NOT_SURE = False  # whether include not sure into the classification

# LABLERS = ["jiale", "peiran", "yuxuan", "huiyu"]
LABLERS = ["jiale", "peiran", "yuxuan"]  # TODO: there is bug when reading huiyu's dataset