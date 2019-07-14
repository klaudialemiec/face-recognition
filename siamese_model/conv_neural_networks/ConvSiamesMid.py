import tensorflow as tf
import numpy as np
import os
from numpy import genfromtxt
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense


def faceRecoModel_Small(input_shape):
    anchor_input = Input(input_shape)
    positive_input = Input(input_shape)
    negative_input = Input(input_shape)

    #build convnet to use in each siamese 'leg'
    convnet = Sequential()
    convnet.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128, (7, 7), activation='relu'))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(128, (4, 4), activation='relu'))
    convnet.add(MaxPooling2D())
    convnet.add(Conv2D(256, (4, 4), activation='relu'))
    convnet.add(Flatten())
    convnet.add(Dense(4096, activation="sigmoid"))

    #call the convnet Sequential siamese_model on each of the input tensors so params will be shared
    encoded_a = convnet(anchor_input)
    encoded_p = convnet(positive_input)
    encoded_n = convnet(negative_input)

    siamese_net = Model(inputs=[anchor_input, positive_input, negative_input], outputs=[encoded_a, encoded_p, encoded_n])

    return siamese_net
