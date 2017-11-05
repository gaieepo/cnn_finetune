import cv2
import numpy as np

import cribriform
from keras import backend as K
from keras.utils import np_utils

nb_train_samples = 1160 # 808+352 training samples pos+neg
nb_valid_samples = 35 # 17+18 validation samples (test)
num_classes = 2

def load_cribriform_data(img_rows, img_cols):

    # Load cribriform training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cribriform.load_data()

    # Resize training images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid
