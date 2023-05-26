from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
import numpy as np


def create_dataset_from_yield_np(dataset):
    from numpy.lib.stride_tricks import sliding_window_view
    u1 = sliding_window_view(dataset["u1"], 820)
    u2 = sliding_window_view(dataset["u2"], 820)
    u3 = sliding_window_view(dataset["u3"], 820)
    return np.concatenate([u1, u2, u3], axis=0)


def yield_csv():
    import os
    import random
    from tqdm import tqdm
    import pandas as pd
    path = "osc_csv/"
    files = os.listdir(path)[:-8]
    random.shuffle(files)
    for file in tqdm(files):
        dataset = pd.read_csv(path + file, index_col="Unnamed: 0")
        yield create_dataset_from_yield_np(dataset) / 320


def total_correlation_loss(z_samples, z_mean, z_log_var):
    # Compute the approximated aggregated posterior, q(z)
    qz = tfp.distributions.Normal(z_mean, tf.exp(0.5 * z_log_var))

    # Compute the approximated factorized prior, product(q(zi))
    qi = tfp.distributions.Normal(z_mean, tf.exp(0.5 * z_log_var))
    log_qi = qi.log_prob(z_samples)

    # Compute the log ratio: log(q(z) / product(q(zi)))
    log_ratio = tf.reduce_sum(log_qi, axis=1) - tf.reduce_sum(qi.log_prob(z_samples), axis=1)
    return tf.reduce_mean(log_ratio) # This is the total correlation.

inputs = Input(820)

c1 = Dense(820, activation='leaky_relu')(inputs)
p1 = Dense(820, activation='leaky_relu')(c1)

c2 = Dense(820, activation='leaky_relu')(p1)
p2 = Dense(820, activation='leaky_relu')(c2)

c3 = Dense(820, activation='leaky_relu')(p2)
p3 = Dense(820, activation='leaky_relu')(c3)

c4 = Dense(820, activation='leaky_relu')(p3)
p4 = Dense(820, activation='leaky_relu')(c4)

c5 = Dense(3, activation='leaky_relu')(p4)

u6 = Dense(3)(c5)
u6 = concatenate([u6, c4])
c6 = Dense(820, activation='leaky_relu')(u6)

u7 = Dense(820, activation='leaky_relu')(c6)
u7 = concatenate([u7, c3])
c7 = Dense(820, activation='leaky_relu')(u7)

u8 = Dense(820)(c7)
u8 = concatenate([u8, c2])
c8 = Dense(820, activation='leaky_relu')(u8)

u9 = Dense(820, activation='leaky_relu')(c8)
u9 = concatenate([u9, c1])
c9 = Dense(820, activation='leaky_relu')(u9)
outputs = Dense(820)(c9)

model = Model(inputs=[inputs], outputs=[outputs])


model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="MSE", metrics="MAE")
# vae.model.load_weights("ae_weights_3.h5")
# model.add_metric(tf.keras.metrics.MAE, name="MAE")
for train in yield_csv():
    model.fit(train, train, epochs=20, validation_split=0.1, batch_size=64, validation_batch_size=64)
    model.model.save_weights("ae_weights_3.h5")


y = model.predict(train)