import cv2
import numpy as np

import cribriform
from keras import backend as K
from keras.utils import np_utils


def load_cribriform_data(img_rows, img_cols, fold):

    num_classes = 2

    # Load cribriform training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cribriform.load_data(fold)
    nb_train_samples = len(Y_train)
    nb_valid_samples = len(Y_valid)

    # Resize training images to 224
    # K.image_dim_ordering() == 'tf'
    X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
    X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])

    # Subtract ImagetNet mean pixel for training and testing data
    X_train[:,:,:,0] -= 103.939
    X_train[:,:,:,1] -= 116.779
    X_train[:,:,:,2] -= 123.68
    X_valid[:,:,:,0] -= 103.939
    X_valid[:,:,:,1] -= 116.779
    X_valid[:,:,:,2] -= 123.68

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid
