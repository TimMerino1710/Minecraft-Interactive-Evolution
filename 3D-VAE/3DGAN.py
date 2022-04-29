import h5py
from time import time
import tensorflow
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, BatchNormalization, Dense, Flatten, Lambda, Reshape, Conv3DTranspose, LeakyReLU, Input, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
from keras.activations import sigmoid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
import os

K.clear_session()


# from https://github.com/enochkan/3dgan-keras/blob/master/models.py
# paper http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf

# =====================
#   GENERATOR
# =====================
def get_generator(latent_dim=200, outdim=32, outchannels=15, kernel_size=(4,4,4), strides=(2,2,2)):

    model = Sequential()
    model.add(Conv3DTranspose(filters=512, kernel_size=kernel_size, strides=(1, 1, 1), kernel_initializer='glorot_normal',bias_initializer='zeros', padding='valid'))
    model.add(BatchNormalization())
    #TODO: change to leaky relu like in the paper
    model.add(Activation('relu'))
    model.add(Conv3DTranspose(filters=256, kernel_size=kernel_size, strides=strides, kernel_initializer='glorot_normal', bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3DTranspose(filters=128, kernel_size=kernel_size, strides=strides, kernel_initializer='glorot_normal', bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # model.add(Conv3DTranspose(filters=64, kernel_size=kernel_size, strides=strides, kernel_initializer='glorot_normal', bias_initializer='zeros', padding='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(Conv3DTranspose(filters=15, kernel_size=kernel_size, strides=strides, kernel_initializer='glorot_normal', bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    noise = Input(shape=(1, 1, 1, latent_dim))
    image = model(noise)
    # for layer in model.layers:
    #     print(layer.output_shape)
    return Model(inputs=noise, outputs=image)

# =====================
#   DISCRIMINATOR
# =====================
def get_discriminator(latent_dim=200, outdim=32, outchannels=15, kernel_size=(4,4,4), strides=(2,2,2)):

    model = Sequential()
    # model.add(Convolution3D(filters=64, kernel_size=kernel_size, strides=strides, kernel_initializer='glorot_normal', bias_initializer='zeros', padding='same'))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution3D(filters=128, kernel_size=kernel_size, strides=strides, kernel_initializer='glorot_normal', bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution3D(filters=256, kernel_size=kernel_size, strides=strides, kernel_initializer='glorot_normal', bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution3D(filters=512, kernel_size=kernel_size, strides=strides, kernel_initializer='glorot_normal', bias_initializer='zeros', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution3D(filters=1, kernel_size=kernel_size, strides=(1, 1, 1), kernel_initializer='glorot_normal', bias_initializer='zeros', padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    image = Input(shape=(outdim, outdim, outdim, outchannels))
    validity = model(image)
    # for layer in model.layers:
    #     print(layer.output_shape)

    return Model(inputs=image, outputs=validity)



d_lr = 1e-5
g_lr = .0025
b1 = .5
batch_size = 256
epochs = 1000
latent_dim = 200
sample_epoch = 100
save_epoch = 500
model_name = 'model_1'
sample_path = "GAN_generated_samples/" + model_name + "/"
checkpoints_path = "GAN_models/" + model_name + "/"

dis_optim = Adam(lr=d_lr, beta_1=b1)
gen_optim = Adam(lr=g_lr, beta_1=b1)

generator = get_generator()
print('Generator')
generator.summary()
z = Input(shape=(1, 1, 1, latent_dim))
img = generator(z)

discriminator = get_discriminator()
print('Discriminator...')
discriminator.summary()
discriminator.compile(loss='binary_crossentropy', optimizer=dis_optim)

# make discriminator not trainable
discriminator.trainable = False
validity = discriminator(img)

combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=gen_optim)

train = np.load('../house_combined_numpy_file/compressedcategorical_combined_rotated_flipped.npy')
# print(train.shape)
# le = LabelEncoder()
# le.fit(train)

train = tf.one_hot(train, 15, dtype=tf.int8).numpy()

# print(np.unique(train[0], axis=0))
# print(np.unique(train[0], axis=0).shape)
# print(len(np.unique(train[0], axis=0)))
# print(train[0])
# print(new.shape)
# enc = OneHotEncoder()
# enc.fit(train)
# print(enc.categories_)
# print(le.classes_)
dl, gl = [],[]

for epoch in range(epochs):
    print("Epoch " + str(epoch) + "...")
    #sample a random batch
    idx = np.random.randint(len(train), size=batch_size)
    real = train[idx]
    # print(len(train))

    z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, latent_dim]).astype(np.float32)
    fake = generator.predict(z)

    # print(real.shape)
    # real = np.expand_dims(real, axis=4)
    # print(real.shape)

    lab_real = np.reshape([1] * batch_size, (-1, 1, 1, 1, 1))
    lab_fake = np.reshape([0] * batch_size, (-1, 1, 1, 1, 1))
    # print(lab_real.shape)

    # calculate discrminator loss
    d_loss_real = discriminator.train_on_batch(real, lab_real)
    d_loss_fake = discriminator.train_on_batch(fake, lab_fake)

    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, latent_dim])#.astype(np.float32)

    # calculate generator loss
    g_loss = combined.train_on_batch(z, np.reshape([1] * batch_size, (-1, 1, 1, 1, 1)))#.astype(np.float64)

    dl.append(d_loss)
    gl.append(g_loss)
    avg_d_loss = round(sum(dl) / len(dl), 4)
    avg_g_loss = round(sum(gl) / len(gl), 4)

    print('Training epoch {}/{}, d_loss_real/avg: {}/{}, g_loss/avg: {}/{}'.format(epoch + 1, epochs, round(d_loss, 4), avg_d_loss, round(g_loss, 4), avg_g_loss))

    # sampling
    if epoch % sample_epoch == 0:
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        print('Sampling...')
        sample_noise = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, latent_dim]).astype(np.float32)
        generated_volumes = generator.predict(sample_noise, verbose=1)
        generated_volumes.dump(sample_path + '/sample_epoch_' + str(epoch + 1) + '.npy')

    # save weights
    if epoch % save_epoch == 0:
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
        generator.save_weights(checkpoints_path + '/generator_epoch_' + str(epoch + 1), True)
        discriminator.save_weights(checkpoints_path + '/discriminator_epoch_' + str(epoch + 1), True)

print("Training finished....")
print('Sampling...')
sample_noise = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, latent_dim]).astype(np.float32)
generated_volumes = generator.predict(sample_noise, verbose=1)
generated_volumes.dump(sample_path + '/sample_final.npy')

generator.save_weights(checkpoints_path + '/generator_final', True)
discriminator.save_weights(checkpoints_path + '/discriminator_final', True)