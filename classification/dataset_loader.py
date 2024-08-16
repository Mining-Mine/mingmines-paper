import argparse
import random
import os

import tensorflow as tf
from collections import Counter
import numpy as np

import config as conf

tf.random.set_seed(42)
random.seed(42)

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

    mine_files = tf.io.gfile.glob(mine_file_pattern)
    mine_dataset_n_samples = len(mine_files)

    # 2. read the images with label = "not_mine"
    not_mine_dir = os.path.join(conf.LABELED_DATASET_PATH, "not_mine")
    not_mine_file_pattern = os.path.join(not_mine_dir, "*.jpg")

    not_mine_files = tf.io.gfile.glob(not_mine_file_pattern)
    not_mine_dataset_n_samples = len(not_mine_files)

    # 3. balance the dataset
    min_n_sample = min([mine_dataset_n_samples, not_mine_dataset_n_samples, conf.MAX_SAMPLES])
    mine_files = random.sample(mine_files, min_n_sample)
    not_mine_files = random.sample(not_mine_files, min_n_sample)
    # label
    mine_labels = [conf.LABEL.MINE] * len(mine_files)
    not_mine_labels = [conf.LABEL.NOT_MINE] * len(not_mine_files)
    # shuffle dataset
    combined_data = list(zip(mine_files, mine_labels)) + list(zip(not_mine_files, not_mine_labels))
    random.shuffle(combined_data)
    shuffled_files, shuffled_labels = zip(*combined_data)
    # form the dataset
    full_dataset = tf.data.Dataset.from_tensor_slices((list(shuffled_files), list(shuffled_labels)))
    full_dataset = full_dataset.map(lambda x, y: (load_image_from_path(x), y))

    # 4. batch the dataset (add batch size handling)
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
    
    def label_extractor(image, label):
        return label
    
    all_labels = dataset.map(label_extractor).batch(total_samples)
    for batch_labels in all_labels.take(1): 
        label_counts.update(batch_labels.numpy().flatten())

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
