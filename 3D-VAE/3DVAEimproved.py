# from https://github.com/kdj842969/3D-Autoencoder


import h5py
from time import time
import tensorflow
from keras.layers import Input
from keras.models import Model
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, BatchNormalization, Dense, Flatten, Lambda, Reshape, Conv3DTranspose
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

def weighted_binary_crossentropy(target, output):
    loss = -(98.0 * target * K.log(output) + 2.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
    return loss

def sampling2(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tensorflow.keras.backend.random_normal(shape=tensorflow.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tensorflow.keras.backend.exp(log_variance/2) * epsilon
    return random_sample


K.clear_session()

num_channels = 15
model_name = 'improved_VAE_1'
sample_path = "VAE_generated_samples/" + model_name + "/"
decoded_path = "VAE_generated_samples/" + model_name + "/decoded/"
model_path = "VAE_models/" + model_name + "/"
epochs = 500

if not os.path.exists(sample_path):
    os.makedirs(sample_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(decoded_path):
    os.makedirs(decoded_path)

data = np.load('../house_combined_numpy_file/20_onehot.npy')
box_size = data.shape[1]
data = data.reshape([-1, box_size, box_size, box_size, num_channels])


train_data, train_rem = train_test_split(data, test_size=0.2, random_state=32)
test_data, val_data = train_test_split(train_rem, test_size=0.5, random_state=32)

z_dim = 400

input_img = Input(shape=(20, 20, 20, 15))
#
#   ENCODER
#
print("========================\nENCODER\n========================")
# does not downsample
x = BatchNormalization()(Convolution3D(256, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer= 'glorot_normal')(input_img))
# downsample to 10x10x10x15
x = BatchNormalization()(Convolution3D(128, (3, 3, 3), strides=(2, 2, 2), activation='elu', padding='same', kernel_initializer= 'glorot_normal')(x))
# does not downsample
x = BatchNormalization()(Convolution3D(64, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer= 'glorot_normal')(x))
# downsample to 5x5x5x15
x = BatchNormalization()(Convolution3D(64, (3, 3, 3), strides=(2, 2, 2), activation='elu', padding='same', kernel_initializer= 'glorot_normal')(x))
# flatten and output
enc_fc1 = BatchNormalization()(Dense(units=1875, kernel_initializer='glorot_normal',activation='elu')(Flatten()(x)))

mu = BatchNormalization()(Dense(units=z_dim, kernel_initializer='glorot_normal', activation=None)(enc_fc1))
sigma = BatchNormalization()(Dense(units=z_dim, kernel_initializer='glorot_normal', activation=None)(enc_fc1))
z = Lambda(sampling2,output_shape = (z_dim, ))([mu, sigma])
encoder = Model(input_img, [mu, sigma, z])

#
#   DECODER
#
print("========================\nDECODER\n========================")
dec_in = Input(shape=(z_dim, ))
x = BatchNormalization()(Dense(1875, kernel_initializer='glorot_normal', activation='elu')(dec_in))
x = Reshape(target_shape=(5,5,5,num_channels))(x)
# does not upsample
x = BatchNormalization()(Conv3DTranspose(64, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal')(x))
# upsamples to 10x10x10x15
x = BatchNormalization()(Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 2), activation='elu', padding='same', kernel_initializer='glorot_normal')(x))
# does not upsample
x = BatchNormalization()(Conv3DTranspose(128, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal')(x))
# upsamples to 20x20x20x15
x = BatchNormalization()(Conv3DTranspose(256, (4, 4, 4), strides=(2, 2, 2), activation='elu', padding='same', kernel_initializer='glorot_normal')(x))
x = BatchNormalization(beta_regularizer=l2(0.001), gamma_regularizer=l2(0.001))(Conv3DTranspose(15, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_initializer='glorot_normal')(x))
decoder = Model(dec_in, x)
decoded = decoder(encoder(input_img)[2])

print("shape of decoded: ")
print(K.int_shape(decoded))

# combined model
autoencoder = Model(input_img, decoded)


voxel_loss = K.cast(K.mean(weighted_binary_crossentropy(input_img, K.clip(sigmoid(decoded), 1e-7, 1.0 - 1e-7))), 'float32') # + kl_div
autoencoder.add_loss(voxel_loss)

learning_rate_1 = 0.0003
learning_rate_2 = 0.0065
momentum = 0.9
sgd = SGD(lr=learning_rate_1, momentum=momentum, nesterov=True)
autoencoder.compile(optimizer=sgd, metrics=['accuracy'])#, loss='categorical_crossentropy')

# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# autoencoder.compile(optimizer='adadelta')
tensorboard = TensorBoard(log_dir="logs\\{}".format(time()))


autoencoder.fit(train_data, train_data,
              epochs=epochs,
              shuffle=True,
              # batch_size=128,
              validation_data=(val_data, val_data),
              callbacks=[tensorboard])

autoencoder.save(model_path + model_name + "/autoencoder.h5")
decoder.save(model_path + model_name + "/decoder.h5")
print("Training finished...")

test_num = test_data.shape[0]
decoded_data = autoencoder.predict(test_data, batch_size=100)
decoded_data = np.argmax(decoded_data, axis=4)
print("Shape of decoded test data: ", decoded_data.shape)
# decoded_data = decoded_data.reshape(test_num, box_size, box_size, box_size)
test_data.dump(decoded_path + 'test.npy')
decoded_data.dump(decoded_path + 'decoded.npy')
np.save("decoded_data_combined_rotated_flipped_1500epoch.npy", decoded_data)

print("testing fininshed")

print("Generating samples:")

num_gen = 100

latent_samples = np.random.normal(size=(num_gen, z_dim))
print(latent_samples.shape)
generated = decoder.predict(latent_samples)
generated = np.argmax(generated, axis=4)
print(generated.shape)
generated.dump(sample_path + 'final.npy')
# for i in range(0, num_gen):
#     np.save('generated_samples/generated1500_' + str(i) + '.npy', generated[i, :, :, :, :])
