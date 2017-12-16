# -*- coding: utf-8 -*-
"""
This is the comment.
"""
import os

import cv2
import numpy as np
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Sequential
from sklearn.cross_validation import train_test_split

__author__ = 'Liushen'

LABEL_DIM = 9
CLASS_DICT = {'classN': 1, 'classM': 2, 'classI': 3, 'classE': 4,
              'classC': 5, 'classPA': 6, 'classA': 7, 'classPX': 8, 'classNO': 0}
BATCH_SIZE = 10
EPOCHS = 20


def custom_model(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(LABEL_DIM, activation='sigmoid'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def get_y(label_list, d):
    result = np.zeros(len(d))
    for l in label_list:
        result[d[l]] = 1.0
    return result


def main():
    x_train_data = []
    y_train_data = []
    with open("image_labels_train.csv", "r") as fd:
        for line in fd:
            filename, labels = line.strip().split(",")
            labels = labels.split("|")
            if os.path.exists(os.path.join("images_train", filename)):
                im = cv2.resize(cv2.imread(os.path.join("images_train", filename)), (224, 224)).astype(np.float32)
                x_train_data.append(im.transpose((2, 0, 1)))
                y_train_data.append(get_y(labels, CLASS_DICT))

    train_x, test_x, train_y, test_y = train_test_split(x_train_data, y_train_data, train_size=0.1, random_state=0)

    x_train = np.array(train_x).astype('float32')
    y_train = np.array(train_y).astype('float32')
    x_test = np.array(test_x).astype('float32')
    y_test = np.array(test_y).astype('float32')

    model = custom_model()
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(x_test, y_test))


if __name__ == '__main__':
    main()
    print("done")
