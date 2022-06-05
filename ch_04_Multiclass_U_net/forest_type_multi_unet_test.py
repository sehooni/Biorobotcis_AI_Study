from forest_type_multi_unet_model import multi_unet_model #Uses softmax 

from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from numpy.testing import assert_allclose

def forest_detection_model(test_img):

    SIZE_X = 256 
    SIZE_Y = 256
    IMG_CHANNELS = 3
    
    n_classes = 6 #Number of classes for segmentation

    def get_model():
        return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=SIZE_Y, IMG_WIDTH=SIZE_X, IMG_CHANNELS=IMG_CHANNELS)

    model = get_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model = load_model("weight_forest_type.h5")

    test_img = cv2.imread(test_img,1)
    dim = (256, 256) #(w,h)

    test_img = cv2.resize(test_img, dim, interpolation=cv2.INTER_AREA)

    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]

    plt.figure(figsize=(12, 8))
    plt.imshow(predicted_img, cmap='jet')
    plt.axis("off")
    plt.savefig('forest_type.png', dpi=960)
    unique, counts = np.unique(predicted_img, return_counts=True)

    return predicted_img, unique, counts

if __name__ == "__main__" :
    test_img = ''
    img, label_unique, label_counts = forest_detection_model(test_img)
