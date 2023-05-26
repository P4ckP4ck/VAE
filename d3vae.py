import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.losses import Loss


class TotalCorrelationLoss(Loss):
    def __init__(self, **kwargs):
        super(TotalCorrelationLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        # Assuming y_pred is the output of a distribution for latent variables
        # and it's shaped as [batch_size, num_latent_vars]

        # Compute joint probability
        joint_prob = K.mean(y_pred, axis=0)

        # Compute marginal probabilities
        marginal_probs = K.prod(y_pred, axis=-1)

        # Compute KL divergence (Total Correlation)
        kl_div = K.sum(joint_prob * K.log(joint_prob / (marginal_probs + K.epsilon())))

        # Compute average Total Correlation
        avg_tc = kl_div / K.cast(K.shape(y_pred)[-1], dtype=K.floatx())

        return avg_tc

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
    for file in tqdm(files[0:1]):
        dataset = pd.read_csv(path + file, index_col="Unnamed: 0")

        yield create_dataset_from_yield_np(dataset) / 320


def kl_divergence(z_mean, z_log_var):
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var) + K.epsilon(), axis=-1)
    return kl_loss


class VAE(tf.keras.Model):
    def __init__(self, model, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.model = model

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            dec_out, z_mu_1, z_sigma_1, z_1, z_mu_2, z_sigma_2, z_2, z_mu_3, z_sigma_3, z_3 = self.model(data)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(data, dec_out))
            kl_1 = kl_divergence(z_mu_1, z_sigma_1)
            kl_2 = kl_divergence(z_mu_2, z_sigma_2)
            kl_3 = kl_divergence(z_mu_3, z_sigma_3)
            kl_loss = (kl_1 + kl_2 + kl_3) / 3
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            "MAE": tf.keras.metrics.MAE(data, dec_out)}



class Sampling(kl.Layer):

    def call(self, args):
        mu, sigma = args
        epsilon = K.random_normal(shape=(K.shape(mu)[0], K.shape(mu)[1]), mean=0., stddev=1.)
        return mu + K.exp(sigma / 2) * epsilon

input_dim = 820
hidden_dim = 410

enc_inp = keras.Input(shape=(input_dim,))
enc_1 = kl.Dense(hidden_dim, activation="leaky_relu")(enc_inp)
enc_2 = kl.Dense(hidden_dim, activation="leaky_relu")(enc_1)
enc_3 = kl.Dense(hidden_dim, activation="leaky_relu")(enc_2)
enc_3 = kl.Flatten()(enc_3)
z_mu_3 = kl.Dense(1, name="z_mu_3")(enc_3)
z_sigma_3 = kl.Dense(1, name="z_sigma_3")(enc_3)
z_3 = Sampling()([z_mu_3, z_sigma_3])

dec_1 = kl.Dense(hidden_dim, activation="leaky_relu")(z_3)
add_2 = kl.Add()([dec_1, enc_2])
add_2 = kl.Flatten()(add_2)
z_mu_2 = kl.Dense(1, name="z_mu_2")(add_2)
z_sigma_2 = kl.Dense(1, name="z_sigma_2")(add_2)
z_2 = Sampling()([z_mu_2, z_sigma_2])

add_4 = kl.Add()([z_2, dec_1])
dec_2 = kl.Dense(hidden_dim, activation="leaky_relu")(add_4)
add_1 = kl.Add()([dec_2, enc_1])
add_1 = kl.Flatten()(add_1)
z_mu_1 = kl.Dense(1, name="z_mu_1")(add_1)
z_sigma_1 = kl.Dense(1, name="z_sigma_1")(add_1)
z_1 = Sampling()([z_mu_1, z_sigma_1])

add_3 = kl.Add()([z_1, dec_2])
dec_out = kl.Dense(input_dim)(add_3)

model = keras.Model(enc_inp, [dec_out, z_mu_1, z_sigma_1, z_1, z_mu_2, z_sigma_2, z_2, z_mu_3, z_sigma_3, z_3], name="vae")
model.summary()
vae = VAE(model)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))
# vae.model.load_weights("ae_weights_3.h5")
vae.model.add_metric(tf.keras.metrics.MAE(enc_inp, dec_out), name="MAE")
for train in yield_csv():
    vae.fit(train, train, epochs=20, validation_split=0.1, batch_size=64, validation_batch_size=64)
    vae.model.save_weights("ae_weights_3.h5")


y = vae.model.predict(train)