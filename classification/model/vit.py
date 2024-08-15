import os

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import config as conf

from tensorflow.keras.applications import vit
from tensorflow.keras import layers, models

class ViT:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = self._build_model()
        print("CNN finish initialization")

    def _build_model(self):
        base_model = vit.ViT(
            include_top=False, 
            weights='imagenet21k', 
            input_shape=conf.INPUT_SHAPE,
            pooling='avg' 
        )

        base_model.trainable = False 

        model = models.Sequential()
        model.add(base_model)

        # Fully connected layer
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))

        # Output layer
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=loss,
                      metrics=['accuracy', 'AUC'])

        print(model.summary())

        return model
