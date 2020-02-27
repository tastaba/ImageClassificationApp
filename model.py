# Import libraries
import os.path as path
import cv2
import numpy as np
import os
import tensorflow as tf
from keras.models import model_from_json
# global graph

class model():
    # def __init__(filePath):

    def process(self,filePath):
        file = cv2.imread(filePath)
        test_image = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
        test_image = cv2.resize(test_image, (200, 200))
        test_image = np.array(test_image)
        test_image = test_image.astype('float32')
        test_image /= 255
        print(test_image.shape)
        test_image = np.expand_dims(test_image, axis=3)
        image = np.expand_dims(test_image, axis=0)
        print(image.shape)
        return image

    def loadModel(self):
        json_file = open('model_cnn.json', 'r')
        self.loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(self.loaded_model_json)
        self.loaded_model.load_weights("model.h10")
        print("Loaded Model from disk")
        self.loaded_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    def predict(self, image):
        self.loadModel()  # have to load model everytime
        # Predicting the class of test images
        probabilities = (self.loaded_model.predict(image))
        print(probabilities)
        confidence = np.nanmax(probabilities)  # nanmax returns the maximum value ina array ignoring the NAN value.
        print(confidence)
        classes = np.argmax(probabilities)  # argmax returns the maximum valued index in an array
        # classes = model.predict_classes(test_image)
        print(classes)
        if classes == 1:
            prediction = 'inattentive'
        else:
            prediction = 'attentive'

        return prediction

    def confidence(self, image):
        self.loadModel()  # have to load model everytime
        # Calculating the confidence of prediction of test images
        probabilities = (self.loaded_model.predict(image))
        confidence = np.nanmax(probabilities)  # nanmax returns the maximum value ina array ignoring the NAN value.
        print(confidence)
        return confidence