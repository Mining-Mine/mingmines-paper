import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from datetime import datetime

import config as conf

tf.random.set_seed(42)


class CNN:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = self._build_model()
        print("CNN finish initialization")

    def _build_model(self):
        model = models.Sequential()

        # Input layer with data augmentation (moved here for illustration, but can be done separately)
        model.add(layers.RandomFlip("horizontal", input_shape=conf.INPUT_SHAPE))
        model.add(layers.RandomRotation(0.1))
        model.add(layers.RandomZoom(0.1))

        # Layer 1
        model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=conf.INPUT_SHAPE))
        model.add(layers.MaxPooling2D((2, 2)))

        # Layer 2
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Layer 3
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        # Flatten and Dense layers
        model.add(layers.Flatten())
        # model.add(layers.Dropout(0.5))
        model.add(layers.Dense(64, activation='relu'))

        # Output layer
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss=loss,
                      metrics=['accuracy', 'AUC'])

        model.summary()  # Optionally keep or remove this in production
        return model

    def load_model(self):
        try:
            self.model.load_weights("./result/cnn/cnn.weights.h5")
            print("loaded model: ./result/cnn/cnn.weights.h5")
        except:
            print("Error loading model: ./result/cnn/cnn.weights.h5")
            exit()

    def train(self, train_data, validation_data=None, batch_size=conf.BATCH_SIZE):
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=conf.TRAIN_EPOCH,
            batch_size=batch_size
        )

        # save the training history
        with open("./result/cnn/train_history.pkl", "wb") as file:
            pickle.dump(history.history, file)

        # save the resulting model
        self.model.save_weights("./result/cnn/cnn.weights.h5")

        return history

    def evaluate_model(self, y_true, y_pred):
        # Calculate the confusion matrix
        cm = confusion_matrix(y_true, np.round(y_pred))
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
        plt.savefig("./result/cnn/CNN_confusion_matrix")

    def plot_samples(self, test_data):
        TP_images = []
        TN_images = []
        FP_images = []
        FN_images = []

        full = 0
        # Classify the images based on the predictions
        for i, (image, label) in enumerate(test_data):
            prediction = np.round(self.predict(image).flatten())

            if label == 1 and prediction == 1:
                if len(TP_images) < 9:
                    TP_images.append(image)  # Append TP images
                    if len(TP_images) >= 9:
                        full += 1
            elif label == 0 and prediction == 0:
                if len(TN_images) < 9:
                    TN_images.append(image)  # Append TN images
                    if len(TN_images) >= 9:
                        full += 1
            elif label == 0 and prediction == 1:
                if len(FP_images) < 9:
                    FP_images.append(image)  # Append FP images
                    if len(FP_images) >= 9:
                        full += 1
            elif label == 1 and prediction == 0:
                if len(FN_images) < 9:
                    FN_images.append(image)  # Append FN images
                    if len(FN_images) >= 9:
                        full += 1
            if full >= 4:
                break

        # Plot a random sample from each category
        self.plot_image_sample(TP_images, "./result/cnn/TP_samples")
        self.plot_image_sample(TN_images, "./result/cnn/TN_samples")
        self.plot_image_sample(FP_images, "./result/cnn/FP_samples")
        self.plot_image_sample(FN_images, "./result/cnn/FN_samples")


    def plot_image_sample(self, images, title):
        plt.figure(figsize=(10, 10))
        for i in range(min(9, len(images))):  # Show up to 9 images
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i][0])
            plt.axis('off')
        plt.suptitle(title)
        plt.savefig(title)
        print("plot samples: {}".format(title))

    def extract_labels(self, dataset):
        labels = [label.numpy() for _, label in dataset]
        return np.array(labels)

    def evaluate_and_plot(self, test_data):
        print("Evaluating on testing set...")
        y_true, y_pred = [], []

        for image, label in test_data:
            y_true.append(label.numpy())
            y_pred.append(self.predict(image).flatten())

        y_true = np.array(y_true)
        y_pred = np.round(np.array(y_pred))

        print("Unique values in y_true:", np.unique(y_true))
        print("Unique values in y_pred:", np.unique(y_pred))
        
        cm = self.evaluate_model(y_true, y_pred)
        self.plot_confusion_matrix(cm)
        self.plot_samples(test_data)

    def predict(self, input_data):

        predictions = self.model.predict(input_data, batch_size=conf.BATCH_SIZE, verbose=None)

        return predictions

    def summary(self):
        self.model.summary()
