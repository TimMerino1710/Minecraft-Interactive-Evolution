# from https://github.com/kdj842969/3D-Autoencoder


import h5py
from time import time
import tensorflow
from keras.layers import Input
from keras.models import Model
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, BatchNormalization, Dense, Flatten, Lambda, Reshape, Conv3DTranspose, Conv3D, Activation, LeakyReLU, Dropout
from keras.regularizers import l2
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
from keras.activations import sigmoid
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf
from keras import backend as K


class VAE_3D():
    def __init__(self):
        K.clear_session()

        self.img_x = 16
        self.img_y = 16
        self.img_z = 16
        self.channels = 11
        self.img_shape = (self.img_x, self.img_y, self.img_z, self.channels)
        self.latent_dim = 500
        self.learning_rate = 0.0006
        self.momentum = .9
        self.model_name = '16_3DVAE_best'
        self.sample_path = "VAE_generated_samples/" + self.model_name + "/"
        self.model_path = "VAE_models/" + self.model_name + "/"
        self.decoded_path = "VAE_generated_samples/" + self.model_name + "/decoded/"
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.decoded_path):
            os.makedirs(self.decoded_path)

        self.load_data()

        self.encoder = self.build_encoder()
        print(self.encoder.summary())
        self.decoder = self.build_decoder()
        print(self.decoder.summary())

        input_img = Input(shape=(self.img_x, self.img_y, self.img_z, self.channels))
        decoded = self.decoder(self.encoder(input_img)[2])
        self.autoencoder = Model(input_img, decoded)

        # the voxel loss is the reconstruction loss
        voxel_loss = K.cast(K.mean(self.weighted_binary_crossentropy(input_img, K.clip(sigmoid(decoded), 1e-7, 1.0 - 1e-7))), 'float32')  # + kl_div
        # kl divergence translated from this lasagne code by original authors brock et al
        # kl_div = -0.5 * K.mean(1 + 2 * Z_ls - T.sqr(Z_mu) - T.exp(2 * Z_ls))
        # in this case, mu is the first output of our encoder and sigma = log variance is the second
        kl_div = -0.5 * K.mean(1 + 2 * self.encoder(input_img)[1] - K.square(self.encoder(input_img)[0]) - K.exp(2 * self.encoder(input_img)[1]))

        total_loss = voxel_loss + kl_div
        # TODO: add regularization term?
        self.autoencoder.add_loss(total_loss)

        self.optimizer = SGD(lr=self.learning_rate, momentum=self.momentum, nesterov=True)

        self.autoencoder.compile(optimizer=self.optimizer, metrics=['accuracy'])


    def weighted_binary_crossentropy(self, target, output):
        loss = -(97.0 * target * K.log(output) + 3.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
        return loss

    def sampling2(mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = tensorflow.keras.backend.random_normal(shape=tensorflow.keras.backend.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + tensorflow.keras.backend.exp(log_variance/2) * epsilon
        return random_sample

    def sampling3(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def build_encoder(self):
        print("========================\nENCODER\n========================")
        input_img = Input(shape=(self.img_x, self.img_y, self.img_z, self.channels))
        # x = BatchNormalization()(Convolution3D(256, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal')(input_img))
        # x = BatchNormalization()(Convolution3D(128, (4, 4, 4), strides=(2, 2, 2), activation='elu', padding='same', kernel_initializer='glorot_normal')(x))
        # x = BatchNormalization()(Convolution3D(64, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal')(x))
        # x = BatchNormalization()(Convolution3D(64, (2, 2, 2), strides=(2, 2, 2), activation='elu', padding='same', kernel_initializer='glorot_normal')(x))
        
        #TODO: restore old
        # x = BatchNormalization()(Convolution3D(128, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.01))(input_img))
        # x = BatchNormalization()(Convolution3D(128, (4, 4, 4), strides=(2, 2, 2), activation='elu', padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.01))(x))
        # x = BatchNormalization()(Convolution3D(64, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.01))(x))
        # x = BatchNormalization()(Convolution3D(32, (4, 4, 4), strides=(2, 2, 2), activation='elu', padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.01))(x))
        x = Conv3D(128, kernel_size=4, strides=1, input_shape=self.img_shape, padding="same")(input_img)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("elu")(x)
        x = Conv3D(54, kernel_size=4, strides=1, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("elu")(x)
        x = Conv3D(64, kernel_size=4, strides=2, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("elu")(x)
        x = Conv3D(128, kernel_size=4, strides=2, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("elu")(x)

        enc_fc1 = BatchNormalization()(Dense(units=4 * 4 * 4 * 32, kernel_initializer='glorot_normal', activation='elu')(Flatten()(x)))

        mu = BatchNormalization()(Dense(units=self.latent_dim, kernel_initializer='glorot_normal', activation=None)(enc_fc1))
        sigma = BatchNormalization()(Dense(units=self.latent_dim, kernel_initializer='glorot_normal', activation=None)(enc_fc1))
        z = Lambda(self.sampling3, output_shape=(self.latent_dim,))([mu, sigma])
        return Model(input_img, [mu, sigma, z])

    def build_decoder(self):
        print("========================\nDECODER\n========================")
        dec_in = Input(shape=(self.latent_dim,))
        
        
        # x = BatchNormalization()(Conv3DTranspose(64, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.01))(x))
        # x = BatchNormalization()(Conv3DTranspose(64, (4, 4, 4), strides=(2, 2, 2), activation='elu', padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.01))(x))
        # x = BatchNormalization()(Conv3DTranspose(128, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.01))(x))
        # x = BatchNormalization()(Conv3DTranspose(128, (4, 4, 4), strides=(2, 2, 2), activation='elu', padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.01))(x))
        
        #TODO: restore old
        # x = BatchNormalization()(Dense(units=4 * 4 * 4 * self.channels, kernel_initializer='glorot_normal', activation='elu')(dec_in))
        # x = Reshape(target_shape=(4, 4, 4, self.channels))(x)
        # x = BatchNormalization()(Convolution3D(64, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.01))(x))
        # x = UpSampling3D()(x)
        # x = BatchNormalization()(Convolution3D(64, (4, 4, 4), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.01))(x))
        # x = BatchNormalization()(Convolution3D(128, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.01))(x))
        # x = UpSampling3D()(x)
        # x = BatchNormalization()(Convolution3D(128, (4, 4, 4), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal', kernel_regularizer=l2(0.01))(x))
        # x = BatchNormalization(beta_regularizer=l2(0.001), gamma_regularizer=l2(0.001))(Convolution3D(self.channels, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_initializer='glorot_normal')(x))
        
        x = Dense(32 * 4 * 4 * 4, activation="relu", input_dim=self.latent_dim)(dec_in)
        x = Reshape((4, 4, 4, 32))(x)
        x = UpSampling3D()(x)
        x = Conv3D(128, kernel_size=4, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("elu")(x)
        x = UpSampling3D()(x)
        x = Conv3D(64, kernel_size=4, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("elu")(x)
        x = Conv3D(64, kernel_size=4, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("elu")(x)
        x = Conv3D(128, kernel_size=4, padding="same")(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation("elu")(x)
        x = Conv3D(self.channels, kernel_size=4, padding="same")(x)
        x = Activation("tanh")(x)

        return Model(dec_in, x)

    def load_data(self):
        self.data = np.load('../house_combined_numpy_file/16_onehot_flipped_smaller_compression.npy')
        box_size = self.data.shape[1]
        self.data = self.data.reshape([-1, box_size, box_size, box_size, self.channels])

        self.train_data, train_rem = train_test_split(self.data, test_size=0.2, random_state=32)
        self.test_data, self.val_data = train_test_split(train_rem, test_size=0.5, random_state=32)

    def train(self, epochs, batch_size):
        self.autoencoder.fit(self.train_data, self.train_data,
                        epochs=epochs,
                        shuffle=True,
                        # batch_size=batch_size,
                        validation_data=(self.val_data, self.val_data))

        self.autoencoder.save(self.model_path +  "/autoencoder.h5")
        self.decoder.save(self.model_path + "/decoder.h5")
        self.encoder.save(self.model_path + "/encoder.h5")

        self.eval_on_test()
        self.generate_samples(50)

    def eval_on_test(self):
        decoded_data = self.autoencoder.predict(self.test_data, batch_size=100)
        decoded_data = np.argmax(decoded_data, axis=4)
        test_categorical =  np.argmax(self.test_data, axis=4)
        test_categorical.dump(self.decoded_path + 'test_data.npy')
        decoded_data.dump(self.decoded_path + 'reconstructed_test_data.npy')

    def generate_samples(self, sample_num):
        latent_samples = np.random.normal(size=(sample_num, self.latent_dim))
        generated = self.decoder.predict(latent_samples)
        generated = np.argmax(generated, axis=4)
        generated.dump(self.sample_path + 'generated_samples.npy')





if __name__ == '__main__':
    vae = VAE_3D()
    vae.train(epochs=500, batch_size=128)