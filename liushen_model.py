# -*- coding: utf-8 -*-
"""
This is the comment.
"""

import argparse
import random

import cv2
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.vgg16 import WEIGHTS_PATH, WEIGHTS_PATH_NO_TOP
from keras.engine import get_source_inputs
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers.core import Flatten, Dense
from keras.models import *
from keras.optimizers import SGD
from keras.utils import layer_utils, get_file
from sklearn.cross_validation import train_test_split

__author__ = 'Liushen'

parser = argparse.ArgumentParser(description="Liushen VGG16 model.")
parser.add_argument("-batch_size", default=128, type=int, help="the batch size")
parser.add_argument("-epochs", default=10, type=int, help="the epochs")
parser.add_argument("-model", default="liushen.sgd.h5", type=str, help="the model file path")
# parser.add_argument("-train_data", default="liushen.train.dat", type=str, help="the train data file path")
parser.add_argument("-test_data_dir", default=None, type=str, help="the test data directory path")
parser.add_argument("-test_data_output", default="liushen.test.txt", type=str, help="the test data output file path")
parser.add_argument("-train_size", default=200000, type=int, help="the train data size")
# parser.add_argument("-gray", default=False, type=bool, help="whether use gray data")
args = parser.parse_args()

LABEL_DIM = 9
IMG_DIM = 512
CLASS_DICT = {'classN': 1, 'classM': 2, 'classI': 3, 'classE': 4,
              'classC': 5, 'classPA': 6, 'classA': 7, 'classPX': 8, 'classNO': 0}
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
MODEL_NAME = args.model
# TRAIN_DATA_FILE = args.train_data
TEST_DATA_DIR = args.test_data_dir
TEST_DATA_OUT_FILE = args.test_data_output
TRAIN_SIZE = args.train_size


def VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=IMG_DIM,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        # x = Dense(classes, activation='softmax', name='predictions')(x) # todo
        x = Dense(LABEL_DIM, activation='sigmoid', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    elif weights is not None:
        model.load_weights(weights)

    return model


def get_y(label_list, d):
    result = np.zeros(len(d))
    for l in label_list:
        result[d[l]] = 1.0
    return result


def get_label(array, d):
    result = {}
    for k in d:
        result[k] = array[d[k]]
    return result


def load_data():
    x_train_data = []
    y_train_data = []
    with open("image_labels_train.csv", "r") as fd:
        for line in fd:
            filename, labels = line.strip().split(",")
            labels = labels.split("|")
            img_filename = os.path.join("images_train", filename)
            if labels[0] == "classNO" and random.random() < 0.65:
                # %65概率跳过反例。。。
                print(img_filename + " skipped.")
                continue
            if os.path.exists(img_filename):
                print(img_filename)
                im = img_file2vector(img_filename)

                # x_train_data.append(im.transpose((2, 0, 1))) # for custom model
                x_train_data.append(im)  # for VGG16
                y_train_data.append(get_y(labels, CLASS_DICT))
                if len(x_train_data) > TRAIN_SIZE:
                    break
    print("all file loaded.")

    train_x, test_x, train_y, test_y = train_test_split(x_train_data, y_train_data, train_size=0.9, random_state=0)
    print("data split.")

    x_train = np.array(train_x).astype('float32')
    y_train = np.array(train_y).astype('float32')
    x_test = np.array(test_x).astype('float32')
    y_test = np.array(test_y).astype('float32')

    return x_train, y_train, x_test, y_test


def img_file2vector(img_filename):
    im = cv2.resize(cv2.imread(img_filename), (IMG_DIM, IMG_DIM)).astype(np.float32)
    im[:, :, 0] -= 128.0
    im[:, :, 1] -= 128.0
    im[:, :, 2] -= 128.0
    return im


def main():
    if not os.path.exists(MODEL_NAME):
        x_train, y_train, x_test, y_test = load_data()

        # model = custom_model()
        model = VGG16(weights=None)
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy')

        print("model to be trained.")
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  verbose=1,
                  validation_data=(x_test, y_test))
        print("model to be saved.")
        model.save(MODEL_NAME)
        print("model saved!")
    else:
        model = load_model(MODEL_NAME)

    if TEST_DATA_DIR is not None:
        test_data = []
        file_list = []
        for each_infile in os.listdir(TEST_DATA_DIR):
            if each_infile.endswith(".png") or each_infile.endswith(".bmp") or each_infile.endswith(".jpg"):
                test_data.append(img_file2vector(os.path.join(TEST_DATA_DIR, each_infile)))
                file_list.append(each_infile)
        x_test = np.array(test_data).astype('float32')
        print("predicting.")
        result = model.predict(x_test, batch_size=BATCH_SIZE)
        print("predicted.")

        with open(TEST_DATA_OUT_FILE, "w") as fd:
            for i, r in enumerate(result):
                label_dict = get_label(r, CLASS_DICT)

                for each_label in label_dict:
                    if label_dict[each_label] > 0.1:
                        fd.write(",".join([file_list[i], each_label, "%.2f" % label_dict[each_label]]) + "\n")


if __name__ == '__main__':
    main()
    print("done")
