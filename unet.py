from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
import numpy as np


def pad_dataset(dataset):
    pad_width = ((0, 0), (0, 2), (2, 0), (0, 0))
    return np.pad(dataset, pad_width, mode='constant', constant_values=0)


inputs = Input((32, 32, 1))

c1 = Conv2D(16, (3, 3), activation='leaky_relu', padding='same')(inputs)
c1 = Conv2D(16, (3, 3), activation='leaky_relu', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='leaky_relu', padding='same')(p1)
c2 = Conv2D(32, (3, 3), activation='leaky_relu', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(p2)
c3 = Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(p3)
c4 = Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation='leaky_relu', padding='same')(p4)
c5 = Conv2D(256, (3, 3), activation='leaky_relu', padding='same')(c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(u6)
c6 = Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(u7)
c7 = Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='leaky_relu', padding='same')(u8)
c8 = Conv2D(32, (3, 3), activation='leaky_relu', padding='same')(c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='leaky_relu', padding='same')(u9)
c9 = Conv2D(16, (3, 3), activation='leaky_relu', padding='same')(c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()