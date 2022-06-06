# https://youtu.be/XyX5HNuv-xE
"""
Author: Dr. Sreenivas Bhattiprolu
Multiclass semantic segmentation using U-Net
Including segmenting large images by dividing them into smaller patches 
and stiching them back
To annotate images and generate labels, you can use APEER (for free):
www.apeer.com 
"""

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

#Resizing images, if needed
SIZE_X = 256
SIZE_Y = 256
n_classes = 6 #Number of classes for segmentation

#Capture training image info as a list
train_images = []

for directory_path in glob.glob("dataset/img_1/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path, 1)
        img = cv2.resize(img, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)
        train_images.append(img)

#Convert list to array for machine learning processing        

train_images = np.array(train_images)
print(train_images.shape)

#Capture mask/label info as a list
train_masks = []

for directory_path in glob.glob("dataset/gt_1/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_masks.append(mask)

#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)
print(train_masks.shape)

###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)
print('before', train_images.shape)
#################################################
# train_images = np.expand_dims(train_images, axis=3)
print('expand', train_images.shape)
train_images = normalize(train_images, axis=1)
print('normarlize', train_images.shape)
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)

#Further split training data t a smaller subset for quick testing of models
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.2, random_state = 0)
print('1', X_train.shape)
print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

###############################################################
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_masks_reshaped_encoded),
                                                 train_masks_reshaped_encoded)
print("Class weights are...:", class_weights)


IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = 3
print(IMG_CHANNELS)
def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# model = load_model("model.h5")

# # define the checkpoint
filepath = 'model.h5'

checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True, verbose=1, mode='min')
callbacks_list =[checkpoint]

# fit the model
model.fit(X_train, y_train_cat, verbose = 1, epochs = 500, batch_size = 32 , validation_data = (X_test, y_test_cat), callbacks = callbacks_list, shuffle=True)

model.save('multicalss_forest_fourth_model.hdf5')
model.save_weights('multiclass_forest_weights.hdf5')
############################################################
#Evaluate the model
	# evaluate model
_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")
