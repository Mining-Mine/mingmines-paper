import random
import os

import config as conf
import dataset_loader as dataset_loader

from model import cnn as cnn
from model import resnet as resnet


if __name__ == '__main__':

    if not os.path.exists(conf.LABELED_DATASET_PATH):
        print("    ERROR: Dataset path not exist: {}".format(conf.LABELED_DATASET_PATH))
        exit(-1)

    # load the dataset
    full_dataset = dataset_loader.load_dataset()
    print("finish loading full dataset")

    # split the training/testing sets
    train_dataset, test_dataset = dataset_loader.split_dataset(full_dataset)

    # init the model
    clf = None
    num_classes = 2
    if conf.INCLUDE_NOT_SURE:
        num_classes += 1

    if conf.MODEL == "CNN":
        clf = cnn.CNN(num_classes = num_classes)
    elif conf.MODEL == "ResNet":
        clf = resnet.ResNet(num_classes = num_classes)
    else:
        print("Models unrecognized!: {}".format(conf.MODEL))
    print("Use Model: {}".format(conf.MODEL))

    # train the model
    if conf.LOAD_SAVED_MODEL:
        clf.load_model()
    else:
        clf.train(train_dataset)
        print("Finish training!")

    # evaluate the model
    clf.evaluate_and_plot(test_dataset)
    print("Finish evaluation!")