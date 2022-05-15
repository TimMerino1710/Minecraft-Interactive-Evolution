import h5py
from time import time
import tensorflow
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, BatchNormalization, Dense, Flatten, Lambda, Reshape, Conv3DTranspose, LeakyReLU, Input, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.constraints import Constraint
from keras import backend
import numpy as np
import tensorflow as tf
from keras import backend as K
import os
import matplotlib.pyplot as plt

K.clear_session()
# K.set_image_dim_ordering('tf')
# K.image_data_format() == 'channels_last'


# from: https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)


class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


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

# from https://github.com/enochkan/3dgan-keras/blob/master/models.py
# paper http://3dgan.csail.mit.edu/papers/3dgan_nips.pdf
# =====================
#   GENERATOR
# =====================
def get_generator(latent_dim=100, outdim=32, outchannels=15, kernel_size=(4,4,4), strides=(2,2,2)):

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
    for layer in model.layers:
        print(layer.output_shape)
    return Model(inputs=noise, outputs=image)

# =====================
#   DISCRIMINATOR
# =====================
def get_critic(latent_dim=100, outdim=32, outchannels=15, kernel_size=(4,4,4), strides=(2,2,2)):
    # weight constraint
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    const = ClipConstraint(.1)
    model = Sequential()
    # model.add(Convolution3D(filters=64, kernel_size=kernel_size, strides=strides, kernel_initializer='glorot_normal', bias_initializer='zeros', padding='same'))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution3D(filters=128, kernel_size=kernel_size, strides=strides, kernel_initializer='glorot_normal', bias_initializer=initializer, padding='same', kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution3D(filters=256, kernel_size=kernel_size, strides=strides, kernel_initializer='glorot_normal', bias_initializer=initializer, padding='same', kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution3D(filters=512, kernel_size=kernel_size, strides=strides, kernel_initializer='glorot_normal', bias_initializer=initializer, padding='same', kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Convolution3D(filters=1, kernel_size=kernel_size, strides=(1, 1, 1), kernel_initializer='glorot_normal', bias_initializer=initializer, padding='valid', kernel_constraint=const))
    model.add(BatchNormalization())
    # model.add(Activation('sigmoid'))
    # WGAN uses linear activation as critic output
    model.add(Dense(1, activation='linear'))
    #
    image = Input(shape=(outdim, outdim, outdim, outchannels))
    # model.summary()
    validity = model(image)
    for layer in model.layers:
        print(layer.output_shape)

    return Model(inputs=image, outputs=validity)




# d_lr = 1e-5
# d_lr = .003
# g_lr = .003
# b1 = .5
# lr=0.00005
lr=0.02
batch_size = 128
epochs = 100
latent_dim = 100
sample_epoch = 1000
save_epoch = 1000
model_name = 'WGAN_model_pointoh2LR'
sample_path = "GAN_generated_samples/" + model_name + "/"
checkpoints_path = "GAN_models/" + model_name + "/"

# dis_optim = Adam(lr=d_lr, beta_1=b1)
# gen_optim = Adam(lr=g_lr, beta_1=b1)
opt = RMSprop(lr=lr)

generator = get_generator()

print('Generator')
generator.summary()
z = Input(shape=(1, 1, 1, latent_dim))
img = generator(z)

critic = get_critic()
print('Discriminator...')
critic.summary()
critic.compile(loss=wasserstein_loss, optimizer=opt)

# make discriminator not trainable
critic.trainable = False
validity = critic(img)

combined = Model(z, validity)
combined.compile(loss=wasserstein_loss, optimizer=opt)

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
dl_r, dl_f = [], []
# number of times we train the critic for every time we train the gen
n_critic = 5
for epoch in range(epochs):
    dl_r_temp, dl_f_temp, dl_tot_temp = [], [], []
    for _ in range(n_critic):
        print("training critic...")
        #sample a random batch
        idx = np.random.randint(len(train), size=batch_size)
        real = train[idx]

        # create fake samples
        z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, latent_dim]).astype(np.float32)
        fake = generator.predict(z)

        # Create labels for reals and fakes. WGAN uses -1 or 1 labels
        lab_real = np.reshape([1.0] * batch_size, (-1, 1, 1, 1, 1))
        lab_fake = np.reshape([-1.0] * batch_size, (-1, 1, 1, 1, 1))

        # train critic, calculate critic loss on reals and fakes
        d_loss_real_temp = critic.train_on_batch(real, lab_real)
        d_loss_fake_temp = critic.train_on_batch(fake, lab_fake)
        dl_r_temp.append(d_loss_real_temp)
        dl_f_temp.append(d_loss_fake_temp)

        d_loss_tot_temp = 0.5 * np.add(d_loss_real_temp, d_loss_fake_temp)
        dl_tot_temp.append(d_loss_tot_temp)

    z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, latent_dim])#.astype(np.float32)
    print("train generator...")
    # calculate generator and critic loss
    g_loss = combined.train_on_batch(z, np.reshape([1.0] * batch_size, (-1, 1, 1, 1, 1)))#.astype(np.float64)
    d_loss = sum(dl_tot_temp) / len(dl_tot_temp)
    d_loss_real = sum(dl_r_temp) / len(dl_r_temp)
    d_loss_fake = sum(dl_f_temp) / len(dl_f_temp)

    # append losts to list to track convergence
    dl_r.append(d_loss_real)
    dl_f.append(d_loss_fake)
    dl.append(d_loss)
    gl.append(g_loss)

    # calculate average critic and generator losses
    avg_d_loss = round(sum(dl) / len(dl), 4)
    avg_g_loss = round(sum(gl) / len(gl), 4)

    print('Training epoch {}/{}, d_loss/avg: {}/{}, d_loss_real: {}, d_loss_fake: {}, g_loss/avg: {}/{}'.format(epoch + 1, epochs, round(d_loss, 4), avg_d_loss, d_loss_real, d_loss_fake, round(g_loss, 4), avg_g_loss))

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
        critic.save_weights(checkpoints_path + '/critic_epoch_' + str(epoch + 1), True)


print("Training finished....")
print("Plotting...")
# plot a graph of our critics loss on reals and fakes, and generator loss
plot_history(dl_r, dl_f, gl, "GAN_models/" + model_name + "/plots", model_name)
print('Sampling...')
sample_noise = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, latent_dim]).astype(np.float32)
generated_volumes = generator.predict(sample_noise, verbose=1)
generated_volumes.dump(sample_path + '/sample_final.npy')

generator.save_weights(checkpoints_path + '/generator_final', True)
critic.save_weights(checkpoints_path + '/critic_final', True)