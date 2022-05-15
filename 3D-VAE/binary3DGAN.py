import h5py
import time
import tensorflow
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, BatchNormalization, Dense, Flatten, Lambda, Reshape, Conv3DTranspose, LeakyReLU, Input, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard
import numpy as np
import tensorflow as tf
from keras import backend as K
import os
import matplotlib.pyplot as plt

K.clear_session()
# K.set_image_dim_ordering('tf')
# K.image_data_format() == 'channels_last'
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def plot_history(d1_hist, d2_hist, g_hist, out_dir, model_name):
    # plot history
    plt.plot(d1_hist, label='crit_real')
    plt.plot(d2_hist, label='crit_fake')
    plt.plot(g_hist, label='gen')
    plt.legend()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(out_dir + '/' + model_name + '_loss.png')
    plt.close()



def build_generator():
    """
    Create a Generator Model with hyperparameters values defined as follows
    :return: Generator network
    """
    z_size = 200
    gen_filters = [512, 256, 128, 64, 32, 1]
    gen_kernel_sizes = [4, 4, 6, 6, 6, 8]
    gen_strides = [1, 1, 2, 2, 2, 1]
    gen_input_shape = (1, 1, 1, z_size)
    gen_activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
    gen_convolutional_blocks = 6

    input_layer = Input(shape=gen_input_shape)

    # First 3D transpose convolution(or 3D deconvolution) block
    a = Conv3DTranspose(filters=gen_filters[0],
        kernel_size=gen_kernel_sizes[0],
        strides=gen_strides[0])(input_layer)
    a = BatchNormalization()(a, training=True)
    a = Activation(activation='relu')(a)

    # Next 4 3D transpose convolution(or 3D deconvolution) blocks
    for i in range(gen_convolutional_blocks - 1):
        a = Conv3DTranspose(filters=gen_filters[i + 1],
            kernel_size=gen_kernel_sizes[i + 1],
            strides=gen_strides[i + 1], padding='same')(a)
        a = BatchNormalization()(a, training=True)
        a = Activation(activation=gen_activations[i + 1])(a)

    gen_model = Model(inputs=input_layer, outputs=a)

    gen_model.summary()
    return gen_model


def build_discriminator():
    """
    Create a Discriminator Model using hyperparameters values defined as follows
    :return: Discriminator network
    """

    dis_input_shape = (32, 32, 32, 1)
    dis_filters = [64, 128, 256, 512, 1]
    dis_kernel_sizes = [6, 6, 6, 4, 4]
    dis_strides = [2, 2, 2, 1, 1]
    dis_paddings = ['same', 'same', 'same', 'same', 'valid']
    dis_alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
    dis_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu',
                       'leaky_relu', 'sigmoid']
    dis_convolutional_blocks = 5

    dis_input_layer = Input(shape=dis_input_shape)

    #TODO: testing with extra layers
    a = Convolution3D(filters=32,
                      kernel_size=8,
                      strides=1,
                      padding='same')(dis_input_layer)
    a = BatchNormalization()(a, training=True)
    a = LeakyReLU(.2)(a)

    # The first 3D Convolutional block
    a = Convolution3D(filters=dis_filters[0],
               kernel_size=dis_kernel_sizes[0],
               strides=dis_strides[0],
               padding=dis_paddings[0])(a)
    a = BatchNormalization()(a, training=True)
    a = LeakyReLU(dis_alphas[0])(a)

    # Next 4 3D Convolutional Blocks
    for i in range(dis_convolutional_blocks - 1):
        a = Convolution3D(filters=dis_filters[i + 1],
                   kernel_size=dis_kernel_sizes[i + 1],
                   strides=dis_strides[i + 1],
                   padding=dis_paddings[i + 1])(a)
        a = BatchNormalization()(a, training=True)
        if dis_activations[i + 1] == 'leaky_relu':
            a = LeakyReLU(dis_alphas[i + 1])(a)
        elif dis_activations[i + 1] == 'sigmoid':
            a = Activation(activation='sigmoid')(a)

    dis_model = Model(inputs=dis_input_layer, outputs=a)
    print(dis_model.summary())
    return dis_model

# gen_learning_rate = 0.0025
# dis_learning_rate = 0.00001
gen_learning_rate = 0.025
dis_learning_rate = 0.01
beta = 0.5
batch_size = 264
epochs = 200
z_size = 200

model_name = 'binaryGAN_biggerkernels_higherDLR'
generated_volumes_dir = "GAN_generated_samples/" + model_name + "/"
if not os.path.exists(generated_volumes_dir):
    os.makedirs(generated_volumes_dir)
# checkpoints_path = "GAN_models/" + model_name + "/"
# DIR_PATH = 'Path to the 3DShapenets dataset directory'
# generated_volumes_dir = 'generated_volumes'
log_dir = 'logs'


# Create instances
generator = build_generator()
discriminator = build_discriminator()

# Specify optimizer
gen_optimizer = Adam(lr=gen_learning_rate, beta_1=beta)
dis_optimizer = Adam(lr=dis_learning_rate, beta_1=0.9)

# Compile networks
generator.compile(loss="binary_crossentropy", optimizer="adam")
discriminator.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

discriminator.trainable = False
adversarial_model = Sequential()
adversarial_model.add(generator)
adversarial_model.add(discriminator)
adversarial_model.compile(loss="binary_crossentropy", optimizer=Adam(lr=gen_learning_rate, beta_1=beta))

tensorboard = TensorBoard(log_dir="{}/{}".format(log_dir, time.time()))
tensorboard.set_model(generator)
tensorboard.set_model(discriminator)

volumes =  np.load('../house_combined_numpy_file/stoneonly_combined_rotated_flipped.npy').astype(float)
gen_hist = []
dis_hist_r = []
dis_hist_f = []
for epoch in range(epochs):
    print("Epoch:", epoch)

    # Create two lists to store losses
    gen_losses = []
    dis_losses = []
    dis_losses_r = []
    dis_losses_f = []

    number_of_batches = int(volumes.shape[0] / batch_size)
    # print("Number of batches:", number_of_batches)
    for index in range(number_of_batches):
        # print("Batch:", index + 1)
        # get a batch of noise vectors and rela volumes
        z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1,z_size]).astype(np.float32)
        volumes_batch = volumes[index * batch_size:(index + 1) * batch_size,:, :, :]

        # use noise vectors to generate fake volumes
        gen_volumes = generator.predict(z_sample, verbose=3)

        # Make the discriminator network trainable
        discriminator.trainable = True

        # Create fake and real labels
        labels_real = np.reshape([1] * batch_size, (-1, 1, 1, 1, 1))
        labels_fake = np.reshape([0] * batch_size, (-1, 1, 1, 1, 1))

        # Train the discriminator network
        loss_real = discriminator.train_on_batch(volumes_batch, labels_real)
        loss_fake = discriminator.train_on_batch(gen_volumes, labels_fake)

        # Calculate total discriminator loss
        d_loss = 0.5 * (loss_real + loss_fake)

        z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)

        # Train the adversarial model
        g_loss = adversarial_model.train_on_batch(z, np.reshape([1] * batch_size, (-1, 1, 1, 1, 1)))

        gen_losses.append(g_loss)
        dis_losses.append(d_loss)
        dis_losses_f.append(loss_fake)
        dis_losses_r.append(loss_real)

        if index  == 0:
            z_sample2 = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
            generated_volumes = generator.predict(z_sample2, verbose=3)
            for i, generated_volume in enumerate(generated_volumes[:5]):
                # print(generated_volume.shape)
                voxels = np.squeeze(generated_volume)
                # print(voxels.shape)
                voxels[voxels < 0.5] = 0.
                voxels[voxels >= 0.5] = 1.
                np.save(generated_volumes_dir + "img_{}_{}_{}".format(epoch, index, i), voxels)
                # saveFromVoxels(voxels, "results/img_{}_{}_{}".format(epoch, index, i))

    gen_hist.append(np.mean(gen_losses))
    dis_hist_r.append(np.mean(dis_losses_r))
    dis_hist_f.append(np.mean(dis_losses_f))
    # writer = TensorBoard.Sum
    print('Training epoch {}/{}, d_loss: {}, g_loss: {}'.format(epoch + 1, epochs, np.mean(dis_losses), np.mean(gen_losses)))
    # write_log(tensorboard, 'g_loss', np.mean(gen_losses), epoch)
    # write_log(tensorboard, 'd_loss', np.mean(dis_losses), epoch)

plot_history(dis_hist_r, dis_hist_f, gen_hist, "GAN_models/" + model_name + "/plots", model_name)
# print("Training finished....")
# print("Plotting...")
# # plot a graph of our critics loss on reals and fakes, and generator loss
# plot_history(dl_r, dl_f, gl, "GAN_models/" + model_name + "/plots", model_name)
# print('Sampling...')
# sample_noise = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, latent_dim]).astype(np.float32)
# generated_volumes = generator.predict(sample_noise, verbose=1)
# generated_volumes.dump(sample_path + '/sample_final.npy')
#
# generator.save_weights(checkpoints_path + '/generator_final', True)
# critic.save_weights(checkpoints_path + '/critic_final', True)