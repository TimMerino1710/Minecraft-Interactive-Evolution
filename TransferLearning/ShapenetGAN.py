# Training a GAN following the architechture and using the dataset (or similar) to Wu et al (https://arxiv.org/pdf/1610.07584.pdf)
# This will be the GAN that is then taken and fine tuned on the minecraft dataset
# We do not have exact code, but will use the same architechture (layers, filter size, strides, loss, etc) as outlined in the paper

# Although they use a straight up 3D-GAN in their paper, we will use a WGAN-GP as it often outperforms basic GANs.

# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division
import sys
sys.path.insert(1, '../3D-VAE/')
from keras.datasets import mnist
from keras.layers.merge import _Merge
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling3D, Conv3D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from functools import partial
import tensorflow as tf
import os
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
from scipy.ndimage import zoom
from visualizer import binary_visualizer



# from https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class ShapenetWGANGP():
    def __init__(self):

        # from paper
        self.img_x = 64
        self.img_y = 64
        self.img_z = 64
        self.channels = 1
        self.img_shape = (self.img_x, self.img_y, self.img_z, self.channels)
        self.latent_dim = 200

        # Storing resutls of this model
        self.model_name = 'shapenetGAN1_tabletest'
        self.sample_path = "GAN_generated_samples_Shapenet/" + self.model_name + "/"
        self.model_path = "GAN_models_Shapenet/" + self.model_name + "/"
        self.visualizer = binary_visualizer(64, 3, 3)
        disable_eager_execution()

        # Create directories for samples and trained models if they don't exist
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # NOT FROM PAPER: WGAN parameters
        # Following parameter and optimizer set as recommended in WGAN-GP paper
        # shapenet paper uses G lr .0025 and D lr 10e-5, back size 100 and Adam with beta=.5
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))

        # Generate images based of noise
        img = self.generator(z_gen)

        # Discriminator determines validity
        valid = self.critic(img)

        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()
        # From paper:
        # "The generator consists of five volumetric fully convolutional layers of kernel sizes 4x4x4 and strides 2, with batch normalization and ReLu layers in between and a sigmoid layer at the end"
        # Note: Batch normalization is ok in the generator, but we will diverge from the papers architecture and not use batch normalization in the critic, as recommended in the WGAN-GP paper https://proceedings.neurips.cc/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf

        model.add(Dense(512 * 4 * 4 * 4, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((4, 4, 4, 512)))
        model.add(UpSampling3D())
        model.add(Conv3D(256, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling3D())
        model.add(Conv3D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling3D())
        model.add(Conv3D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling3D())
        model.add(Conv3D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("sigmoid"))

        # TODO: This is technically only 4 convolutional layers. They say they have 5 in the paper, but their diagram doesnt really reflect that because of the first layer, I believe (page 3)
        # TODO: "there are no pooling or linear layers in our network", does that mean my first layer is wrong? Its just a dense connected to the latent

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        # From paper:
        # "The discriminator basically mirrors the generator, except that is uses LeakyReLU instead of ReLU layers. There are no pooling or linera layers in our network

        model.add(Conv3D(64, kernel_size=4, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        # downsampled to 64x32x32x32
        model.add(Conv3D(128, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        # downsampled to 128x16x16x16
        model.add(Conv3D(256, kernel_size=4, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        # downsampled to 256x8x8x8
        model.add(Conv3D(512, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        # downsampled to 512x4x4x4
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def load_data(self):
        # modelnet:
        # data = np.load('modelnet_64_scaled.npy')

        # shapenet:
        data = np.load('shapenet_table_combined.npy')
        X = data.reshape([-1, 64, 64, 64, 1])
        return X

    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        X_train = self.load_data()
        self.sample_reals(X_train[:9])

        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0 or epoch == epochs - 1:
                self.sample_images(epoch)
                self.generator.save(self.model_path + "/generator_" + str(epoch))
                self.critic.save(self.model_path + "/critic_" + str(epoch))
                self.generator_model.save(self.model_path + "/GAN_" + str(epoch))
                config = self.generator_model.get_config()


        self.generator.save(self.model_path + "/generator_final.h5")


    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs[gen_imgs >= .5] = 1
        gen_imgs[gen_imgs < .5] = 0
        # gen_imgs = np.argmax(gen_imgs, axis=4)
        gen_imgs.dump(self.sample_path + '/sample_epoch_' + str(epoch + 1) + '.npy')

        self.visualizer.draw(gen_imgs, self.sample_path + '/figures_epoch_' + str(epoch) + '.png')
        # # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5
        #
        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
        #         axs[i,j].axis('off')
        #         cnt += 1
        # fig.savefig("images/mnist_%d.png" % epoch)
        # plt.close()

    def sample_reals(self, data):
        self.visualizer.draw(data, self.sample_path + '/figures_reals' + '.png')


if __name__ == '__main__':
    wgan = ShapenetWGANGP()
    wgan.train(epochs=100, batch_size=32, sample_interval=199)