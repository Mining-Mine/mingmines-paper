import os

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import config as conf

class CNN:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = self._build_model()
        print("CNN finish initialization")


    def _build_model(self):
        model = models.Sequential()

        # layer 1
        model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=conf.INPUT_SHAPE))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

        # layer 2
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # layer 3
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Flatten layer
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))

        # Fully connected layer - 减少全连接层的神经元数量
        model.add(layers.Dense(64, activation='relu'))

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

    def train(self, train_data, validation_data=None, batch_size=conf.BATCH_SIZE):
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=conf.TRAIN_EPOCH,
            batch_size=batch_size
        )
        return history

    def evaluate_model(self, y_true, y_pred):
        # Calculate the confusion matrix
        cm = confusion_matrix(y_true, np.round(y_pred))
        print("Confusion matrix shape:", cm.shape)
        TP, FP, FN, TN = cm.ravel()

        # Calculate various metrics
        accuracy = (TP + TN) / float(TP + TN + FP + FN)
        precision = TP / float(TP + FP)
        TPR = recall = TP / float(TP + FN)  # Recall
        TNR = TN / float(TN + FP)
        FPR = FP / float(TN + FP)
        FNR = FN / float(TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"TPR (Recall): {TPR}")
        print(f"TNR: {TNR}")
        print(f"FPR: {FPR}")
        print(f"FNR: {FNR}")
        print(f"F1 Score: {f1}")

        return cm

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("./CNN_confusion_matrix")

    def plot_samples(self, test_data, y_true, y_pred):
        TP_images = []
        TN_images = []
        FP_images = []
        FN_images = []

        # Classify the images based on the predictions
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                TP_images.append(test_data[i][0])  # Append TP images
            elif y_true[i] == 0 and y_pred[i] == 0:
                TN_images.append(test_data[i][0])  # Append TN images
            elif y_true[i] == 0 and y_pred[i] == 1:
                FP_images.append(test_data[i][0])  # Append FP images
            elif y_true[i] == 1 and y_pred[i] == 0:
                FN_images.append(test_data[i][0])  # Append FN images

        # Plot a random sample from each category
        self.plot_image_sample(TP_images, "TP_samples")
        self.plot_image_sample(TN_images, "TN_samples")
        self.plot_image_sample(FP_images, "FP_samples")
        self.plot_image_sample(FN_images, "FN_samples")

    def plot_image_sample(self, images, title):
        plt.figure(figsize=(10, 10))
        for i in range(min(9, len(images))):  # Show up to 9 images
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.axis('off')
        plt.suptitle(title)
        plt.savefig(title)
        print("plot samples: {}".format(title))

    def extract_images_and_labels(self, dataset):
        images = []
        labels = []
        for image, label in dataset:
            images.append(image.numpy())
            labels.append(label.numpy())
        return np.array(images), np.array(labels)

    def evaluate_and_plot(self, test_data):
        images, y_true = self.extract_images_and_labels(test_data)
        y_pred = self.predict(test_data)

        print("num of classes = {}".format(self.num_classes))
        if self.num_classes == 2:
            y_pred = np.round(y_pred).flatten()  # Round off for binary classification

        # Evaluate the model and print results
        print("Unique values in y_true:", np.unique(y_true))
        print("Unique values in y_pred:", np.unique(y_pred))
        cm = self.evaluate_model(y_true, y_pred)

        # Plot the confusion matrix
        self.plot_confusion_matrix(cm)

        # Plot sample images for TP, TN, FP, FN
        self.plot_samples(images, y_true, y_pred)

    def predict(self, input_data):
        predictions = self.model.predict(input_data, batch_size=conf.BATCH_SIZE)
        return predictions

    def summary(self):
        self.model.summary()
