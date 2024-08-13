import random
import os

import tensorflow as tf

import config as conf
import dataset_loader as dataset_loader


if __name__ == '__main__':

    if not os.path.exists(conf.LABELED_DATASET_PATH):
        print("    ERROR: Dataset path not exist: {}".format(conf.LABELED_DATASET_PATH))
        exit(-1)

    full_dataset = dataset_loader.load_dataset()
    dataset_loader.dataset_summary(full_dataset)