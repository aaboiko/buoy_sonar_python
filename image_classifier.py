import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.models import Sequential
import cv2
import os

class ImageClassifier:
    def __init__(self) -> None:
        self.cnn = self.create_cnn()

    def create_cnn(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10))

        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        return model
    

    def fit_cnn(self, x_train, y_train, x_test, y_test, epochs):
        history = self.cnn.fit(x_train, y_train, epochs=epochs, 
                    validation_data=(x_test, y_test))
        
        return history