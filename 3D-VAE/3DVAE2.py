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

import tensorflow as tf
from keras import backend as K

def weighted_binary_crossentropy(target, output):
    loss = -(98.0 * target * K.log(output) + 2.0 * (1.0 - target) * K.log(1.0 - output)) / 100.0
    return loss

def sampling(args):
    mu, sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape = (batch, dim))

    return mu + K.exp(0.5 * sigma) * epsilon

def sampling2(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tensorflow.keras.backend.random_normal(shape=tensorflow.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tensorflow.keras.backend.exp(log_variance/2) * epsilon
    return random_sample

# def voxel_loss(inputs, outputs):
#     voxel_loss = K.cast(K.mean(weighted_binary_crossentropy(inputs, K.clip(sigmoid(outputs), 1e-7, 1.0 - 1e-7))), 'float32')  # + kl_div
#     return voxel_loss


K.clear_session()
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.experimental.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available())
# with h5py.File('object.hdf5', 'r') as f:
#     train_data = f['train_mat'][...]
#     val_data = f['val_mat'][...]
#     test_data = f['test_mat'][...]

# train_data = np.load('../house_combined_numpy_file/combined.npy')
data = np.load('../house_combined_numpy_file/stoneonly_combined_rotated_flipped.npy')
train_data, train_rem = train_test_split(data, test_size=0.2, random_state=32)
test_data, val_data = train_test_split(train_rem, test_size=0.5, random_state=32)

print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

train_num = train_data.shape[0]
val_num = val_data.shape[0]
test_num = test_data.shape[0]
box_size = train_data.shape[1]

train_data = train_data.reshape([-1, box_size, box_size, box_size, 1])
val_data = val_data.reshape([-1, box_size, box_size, box_size, 1])
test_data = test_data.reshape([-1, box_size, box_size, box_size, 1])

print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

z_dim = 128

input_img = Input(shape=(32, 32, 32, 1))
#
#   ENCODER
#
print("========================\nENCODER\n========================")
#TODO: testing from paper params aka 3DVAE.py
x = BatchNormalization()(Convolution3D(8, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='valid', kernel_initializer= 'glorot_normal')(input_img))
print(K.int_shape(x))
x = BatchNormalization()(Convolution3D(16, (3, 3, 3), strides=(2, 2, 2), activation='elu', padding='same', kernel_initializer= 'glorot_normal')(x))
print(K.int_shape(x))
x = BatchNormalization()(Convolution3D(32, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='valid', kernel_initializer= 'glorot_normal')(x))
print(K.int_shape(x))
x = BatchNormalization()(Convolution3D(64, (3, 3, 3), strides=(2, 2, 2), activation='elu', padding='same', kernel_initializer= 'glorot_normal')(x))
print(K.int_shape(x))
enc_fc1 = BatchNormalization()(Dense(units=343, kernel_initializer='glorot_normal',activation='elu')(Flatten()(x)))
mu = BatchNormalization()(Dense(units=z_dim, kernel_initializer='glorot_normal', activation=None)(enc_fc1))
sigma = BatchNormalization()(Dense(units=z_dim, kernel_initializer='glorot_normal', activation=None)(enc_fc1))
z = Lambda(sampling2,output_shape = (z_dim, ))([mu, sigma])
encoder = Model(input_img, [mu, sigma, z])

# TODO: uncommment for original
# # x = Convolution3D(10, (5, 5, 5), activation='relu', padding='same')(input_img)
# # x = MaxPooling3D((2, 2, 2), padding='same')(x)
# x = Convolution3D(30, (5, 5, 5), activation='relu', padding='same')(input_img)
# x = MaxPooling3D((2, 2, 2), padding='same')(x)
# x = Convolution3D(60, (5, 5, 5), activation='relu', padding='same')(x)
# encoded = MaxPooling3D((2, 2, 2), padding='same')(x)
# # x = Convolution3D(120, (5, 5, 5), activation='relu', padding='same')(x)
# # encoded = MaxPooling3D((2, 2, 2), padding='same', name='encoder')(x)

print("shape of encoded: ")
# print(K.int_shape(encoded))
print(z_dim)
#
#   DECODER
#
print("========================\nENCODER\n========================")
#TODO: testing from paper params aka 3DVAE.py
dec_in = Input(shape=(z_dim, ))
x = BatchNormalization()(Dense(343, kernel_initializer='glorot_normal', activation='elu')(dec_in))
x = Reshape(target_shape = (7,7,7,1))(x)
print(K.int_shape(x))
x = BatchNormalization()(Conv3DTranspose(64, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal')(x))
print(K.int_shape(x))
x = BatchNormalization()(Conv3DTranspose(32, (3, 3, 3), strides=(2, 2, 2), activation='elu', padding='valid', kernel_initializer='glorot_normal')(x))
print(K.int_shape(x))
x = BatchNormalization()(Conv3DTranspose(16, (3, 3, 3), strides=(1, 1, 1), activation='elu', padding='same', kernel_initializer='glorot_normal')(x))
print(K.int_shape(x))
x = BatchNormalization()(Conv3DTranspose(8, (4, 4, 4), strides=(2, 2, 2), activation='elu', padding='valid', kernel_initializer='glorot_normal')(x))
print(K.int_shape(x))
x = BatchNormalization(beta_regularizer=l2(0.001), gamma_regularizer=l2(0.001))(Conv3DTranspose(1, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_initializer='glorot_normal')(x))
decoder = Model(dec_in, x)
decoded = decoder(encoder(input_img)[2])

# TODO: uncommment for original
# # x = Convolution3D(16, (5, 5, 5), activation='relu', padding='same')(encoded)
# # x = UpSampling3D((2, 2, 2))(x)
# x = Convolution3D(60, (5, 5, 5), activation='relu', padding='same')(encoded)
# x = UpSampling3D((2, 2, 2))(x)
# x = Convolution3D(30, (5, 5, 5), activation='relu', padding='same')(x)
# x = UpSampling3D((2, 2, 2))(x)
# # x = Convolution3D(10, (5, 5, 5), activation='relu', padding='same')(input_img)
# # x = MaxPooling3D((2, 2, 2), padding='same')(x)
# decoded = Convolution3D(1, (5, 5, 5), activation='relu', padding='same')(x)
print("shape of decoded: ")
print(K.int_shape(decoded))



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
              epochs=1600,
              shuffle=True,
              # batch_size=128,
              validation_data=(val_data, val_data),
              callbacks=[tensorboard])

autoencoder.save('variationalautoencoder_flipped_1500epoch.h5')
decoder.save('decoder_flipped_1500epoch.h5')
print("Training finished...")

decoded_data = autoencoder.predict(test_data, batch_size=100)
decoded_data = decoded_data.reshape(test_num, box_size, box_size, box_size)
np.save("decoded_data_combined_rotated_flipped_1500epoch.npy", decoded_data)

for i in range(0, 20):
    struct = test_data[i]
    recon_struct = decoded_data[i]
    if i == 0:
        print("decoded shape: ", recon_struct.shape)
    # np.save('decoded_test_data/2test_reconstructed_' + str(i), recon_struct)
    # np.save('decoded_test_data/2test_' + str(i), struct)
print("testing fininshed")

print("Generating samples:")

num_gen = 20

latent_samples = np.random.rand(num_gen, z_dim)
generated = decoder.predict(latent_samples)
print(generated.shape)
for i in range(0, num_gen):
    np.save('generated_samples/generated1500_' + str(i) + '.npy', generated[i, :, :, :, :])
