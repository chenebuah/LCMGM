# CONTRIBUTORS: * Ericsson Chenebuah, Michel Nganbe and Alain Tchagang 
# Department of Mechanical Engineering, University of Ottawa, 75 Laurier Ave. East, Ottawa, ON, K1N 6N5 Canada
# Digital Technologies Research Centre, National Research Council of Canada, 1200 Montréal Road, Ottawa, ON, K1A 0R6 Canada
# * email: echen013@uottawa.ca 
# (December-2023)

## THIS CODE EXECUTES THE SEMI-SUPERVISORY FRAMEWORK FOR TARGET-LEARNING THE FORMATION ENERGY PARAMETER AND CRYSTAL SYSTEM TYPE IN THE LATENT SPACE OF A VARIATIONAL AUTOENCODER.
# PHASE 1: SEMI-SUPERVISORY VARIATIONAL AUTOENCODER (SS-VAE) MODEL

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Conv2DTranspose, LeakyReLU, Activation, Flatten, Reshape, BatchNormalization, Conv2D
from keras import backend as K
from keras.models import Model
from tensorflow.keras import layers

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0, stddev=1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(32,32,3))
x = Conv2D(32, (3, 3), activation=LeakyReLU(alpha=0.2), strides= 2, padding='same')(encoder_inputs)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation=LeakyReLU(alpha=0.2), strides= 2, padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation=LeakyReLU(alpha=0.2), strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dense(1024, activation="sigmoid")(x)
z_mean = Dense(latent_dim, name="z_mean")(x)
z_log_var = Dense(latent_dim, name="z_log_var")(x)
z = Lambda(Sampling(), output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Formation Energy Regressor MLP
reg_latent_inputs = Activation("relu")(z)
x = Dense(256, activation="relu")(reg_latent_inputs)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
reg_outputs = Dense(1, activation='linear', name='reg_output')(x)
reg_supervised = Model(reg_latent_inputs, reg_outputs, name='reg')

# Crystal Class Classifier MLP
clf_latent_inputs = Activation("relu")(z)
x = Dense(128, activation="relu")(clf_latent_inputs)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
clf_outputs = Dense(5, activation='softmax', name='class_output')(x)
clf_supervised = Model(clf_latent_inputs, clf_outputs, name='clf')

# Decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(1024, activation="sigmoid")(latent_inputs)
x = Dense((32/4 * 32/4 * 128), activation=LeakyReLU(alpha=0.2))(x)
x = BatchNormalization()(x)
x = Reshape((int(32/4), int(32/4), 128))(x)
x = Conv2DTranspose(64, (3, 3), activation=LeakyReLU(alpha=0.2), strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(32, (3, 3), activation=LeakyReLU(alpha=0.2), strides=2, padding='same')(x)
x = BatchNormalization()(x)
decoder_outputs = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same', strides=2)(x)
decoder = Model(latent_inputs, decoder_outputs, name="decoder")

# SS-VAE Compile
outputs = [decoder(encoder(encoder_inputs)[2]), reg_supervised(encoder(encoder_inputs)[2]), 
           clf_supervised(encoder(encoder_inputs)[2])]
vae = Model(encoder_inputs, outputs, name='vae_mlp')
reconstruction_loss = (tf.reduce_mean(tf.reduce_sum(K.square(encoder_inputs - outputs[0]), axis=[1,2])))*32*32*3
kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
vae_loss = reconstruction_loss + kl_loss
vae.add_loss(vae_loss)
vae.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4, decay=1e-4 / 200),
    loss={'clf': 'categorical_crossentropy', 'reg': 'mean_squared_error'},
    metrics={'clf': 'accuracy', 'reg': 'mae'}
            )

# Train SS-VAE model
svae_history = vae.fit(X_samples, {'clf': y_cs_, 'reg': y_Ef},  epochs=500)


# Encoded Latent Vectors
_, _, z = encoder.predict(X_samples)
