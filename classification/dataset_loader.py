import argparse
import random
import os

import tensorflow as tf
from collections import Counter
import numpy as np

import config as conf

tf.random.set_seed(42)

def load_image_from_path(path):
    image = tf.io.read_file(path)
    try:
        image = tf.image.decode_jpeg(image, channels=conf.INPUT_SHAPE[2]) # num of channels
    except tf.errors.InvalidArgumentError:
        print("Error decoding image: {}".format(path))
        exit()
        return None, None
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Resize the image to the target size
    image = tf.image.resize(image, (conf.INPUT_SHAPE[0], conf.INPUT_SHAPE[1]))
    
    return image

def filter_valid_images(image, label):
    return image is not None

def load_dataset():
    full_dataset = None

    # 1. read the images with label = "mine"
    mine_dir = os.path.join(conf.LABELED_DATASET_PATH, "mine")
    mine_file_pattern = os.path.join(mine_dir, "*.jpg")

    mine_dataset = tf.data.Dataset.list_files(mine_file_pattern)
    mine_dataset = mine_dataset.map(lambda x: (load_image_from_path(x), conf.LABEL.MINE))
    mine_dataset_n_samples = tf.data.experimental.cardinality(mine_dataset).numpy()

    # 2. read the images with label = "not_mine"
    not_mine_dir = os.path.join(conf.LABELED_DATASET_PATH, "not_mine")
    not_mine_file_pattern = os.path.join(not_mine_dir, "*.jpg")

    not_mine_dataset = tf.data.Dataset.list_files(not_mine_file_pattern) 
    not_mine_dataset = not_mine_dataset.map(lambda x: (load_image_from_path(x), conf.LABEL.NOT_MINE))
    not_mine_dataset_n_samples = tf.data.experimental.cardinality(not_mine_dataset).numpy()

    # balance the dataset
    min_n_sample = min([mine_dataset_n_samples, not_mine_dataset_n_samples])
    mine_dataset = mine_dataset.take(min_n_sample)
    not_mine_dataset = not_mine_dataset.take(min_n_sample)

    combined_dataset = mine_dataset.concatenate(not_mine_dataset)
    # Initialize or concatenate with full_dataset
    if full_dataset is None:
        full_dataset = combined_dataset
    else:
        full_dataset = full_dataset.concatenate(combined_dataset)

    if full_dataset is None:
        raise ValueError("No datasets were loaded. Please check the dataset path.")

    # 4. shuffle the dataset
    full_dataset = full_dataset.shuffle(buffer_size=1000)

    # 5. batch the dataset (add batch size handling)
    batch_size = conf.BATCH_SIZE if hasattr(conf, 'BATCH_SIZE') else 32
    full_dataset = full_dataset.batch(batch_size)

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
        for l in label.numpy():
            label_counts[l.item()] += 1

    print("\nDataset Summary:")
    print(f"Total number of samples: {total_samples}")
    print(f"Label distribution: {label_counts}")
    print(f"Element shapes (first {num_samples} samples): {element_shapes}")
    print(f"Unique shapes: {set(element_shapes)}")
    print(f"Sample labels: {labels}")
    print(f"Element spec: {dataset.element_spec}")


def split_dataset(full_dataset, train_size=0.7):
    dataset_size = full_dataset.cardinality().numpy()
    
    train_size = int(train_size * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    
    return train_dataset, test_dataset
