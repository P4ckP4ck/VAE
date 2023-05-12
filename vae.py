import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose, Reshape, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.cluster import KMeans
import tensorflow_probability as tfp


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


# Network parameters
intermediate_dim = 100
latent_dim = 3
batch_size = 100
epochs = 50

# Encoder
# Input shape (height, width, channels)
input_shape = (820,1)

# Encoder architecture
inputs = Input(shape=input_shape)

x = Conv1D(410, kernel_size=5, activation='leaky_relu')(inputs)

x = Conv1D(205, kernel_size=5, activation='leaky_relu')(inputs)
x = Conv1D(intermediate_dim, kernel_size=5, activation='leaky_relu')(inputs)
h = Flatten()(x)

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return K.in_train_phase(z_mean + K.exp(z_log_var / 2) * epsilon, z_mean)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
decoder_h = Dense(intermediate_dim, activation='leaky_relu')(z)
decoder_1 = Dense(200, activation='leaky_relu')(decoder_h)
decoder_2 = Dense(410, activation='leaky_relu')(decoder_1)
decoder_out = Dense(820, activation="linear")(decoder_2)

pdec_inp = Input(3)
pdecoder_h = Dense(intermediate_dim, activation='leaky_relu')(pdec_inp)
pdecoder_1 = Dense(200, activation='leaky_relu')(pdecoder_h)
pdecoder_2 = Dense(410, activation='leaky_relu')(pdecoder_1)
pdecoder_out = Dense(820, activation="linear")(pdecoder_2)


# VAE model
vae = Model(inputs, decoder_out)

# VAE loss

# reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(decoder_out)) * np.prod(input_shape)
reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.squeeze(inputs), decoder_out)) * np.prod(input_shape)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var + K.epsilon()), axis=-1)

vae_loss = K.mean(reconstruction_loss + kl_loss + total_correlation_loss(z, z_mean, z_log_var))

vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(learning_rate=0.001), metrics=["MSE"])
# vae.load_weights("ae_weights.h5")

for train in yield_csv():
    residual = len(train) % 32
    if residual != 0:
        train = train[:-residual]
    # break
    vae.fit(train, train, epochs=1, validation_split=0.1)
    vae.save_weights("ae_weights.h5")

# Encoder model

encoder = Model(inputs, [z_mean, z_log_var])
sample_encoder = Model(inputs, z)
decoder = Model(decoder_h, decoder_out)
z_pred = encoder.predict(train[0:100*32])
zs_pred = sample_encoder.predict(train[0:100*32])
# y = decoder.predict(z_pred[0])
x_pred = vae.predict(train[0:100*32])


