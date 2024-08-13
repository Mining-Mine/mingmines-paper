import argparse
import random
import os

import tensorflow as tf
from collections import Counter
import numpy as np

import config as conf

def load_image_from_path(path):
    image = tf.io.read_file(path)
    try:
        image = tf.image.decode_jpeg(image, channels=3)
    except tf.errors.InvalidArgumentError:
        print("Error decoding image: {}".format(path))
        exit()
        return None, None
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def filter_valid_images(image, label):
    return image is not None

def load_dataset():
    full_dataset = None

    # load the dataset from each labler
    for labler in conf.LABLERS:
        print("load images from: {}".format(labler))
        labler_dir = os.path.join(conf.LABELED_DATASET_PATH, labler)

        # 1. read the images with label = "mine"
        mine_dir = os.path.join(labler_dir, "mine")
        mine_file_pattern = os.path.join(mine_dir, "*.jpg")

        mine_dataset = tf.data.Dataset.list_files(mine_file_pattern)
        mine_dataset = mine_dataset.map(lambda x: (load_image_from_path(x), conf.LABEL.MINE))

        # 2. read the images with label = "not_mine"
        not_mine_dir = os.path.join(labler_dir, "not_mine")
        not_mine_file_pattern = os.path.join(not_mine_dir, "*.jpg")

        not_mine_dataset = tf.data.Dataset.list_files(not_mine_file_pattern) 
        not_mine_dataset = not_mine_dataset.map(lambda x: (load_image_from_path(x), conf.LABEL.NOT_MINE))

        # 3. [Option] read the images with label = "not_sure"
        if conf.INCLUDE_NOT_SURE:
            not_sure_dir = os.path.join(labler_dir, "not_sure")
            not_sure_file_pattern = os.path.join(not_sure_dir, "*.jpg")

            not_sure_dataset = tf.data.Dataset.list_files(not_sure_file_pattern)
            not_sure_dataset = not_sure_dataset.map(lambda x: (load_image_from_path(x), conf.LABEL.NOT_SURE))

            # Concatenate all datasets
            combined_dataset = mine_dataset.concatenate(not_mine_dataset).concatenate(not_sure_dataset)
        else:
            # Concatenate mine and not_mine datasets
            combined_dataset = mine_dataset.concatenate(not_mine_dataset)

        # Initialize or concatenate with full_dataset
        if full_dataset is None:
            full_dataset = combined_dataset
        else:
            full_dataset = full_dataset.concatenate(combined_dataset)

    if full_dataset is None:
        raise ValueError("No datasets were loaded. Please check LABLERS and paths.")

    # 4. shuffle the dataset
    full_dataset = full_dataset.shuffle(buffer_size=2000)

    return full_dataset


def dataset_summary(dataset, num_samples=5):
    element_shapes = []
    labels = []

    print(dataset)
    
    for image, label in dataset.take(num_samples):
        element_shapes.append(image.shape)
        labels.append(label.numpy())
        print(f"Sample image shape: {image.shape}, Label: {label.numpy()}")

    total_samples = tf.data.experimental.cardinality(dataset).numpy()

    label_counts = Counter()
    for _, label in dataset:
        label_counts[label.numpy()] += 1

    print("\nDataset Summary:")
    print(f"Total number of samples: {total_samples}")
    print(f"Label distribution: {label_counts}")
    print(f"Element shapes (first {num_samples} samples): {element_shapes}")
    print(f"Unique shapes: {set(element_shapes)}")
    print(f"Sample labels: {labels}")
    print(f"Element spec: {dataset.element_spec}")