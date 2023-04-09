import sys
sys.path.insert(1, '../3D-VAE/')
from keras.models import load_model
import keras.backend as K
import numpy as np
import os
from tensorflow.keras.layers import Input
from visualizer import binary_visualizer
from tensorflow.keras.models import Sequential, Model
from functools import partial
from tensorflow.keras.optimizers import RMSprop
from keras.layers.merge import _Merge
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution

GENERATOR_PATH = 'C:\\Users\\timme\\Documents\\MinecraftStructurePCG\\TransferLearning\\GAN_models_Shapenet\\shapenetGAN1_table4\\generator_2999'
CRITIC_PATH = 'C:\\Users\\timme\\Documents\\MinecraftStructurePCG\\TransferLearning\\GAN_models_Shapenet\\shapenetGAN1_table4\\critic_2999'
GAN_PATH = 'C:\\Users\\timme\\Documents\\MinecraftStructurePCG\\TransferLearning\\GAN_models_Shapenet\\shapenetGAN1_table4\\GAN_99'


MODEL_NAME = "transfer_learning_test1_nonefrozen_tinylr"
SAMPLE_PATH = "H:\\Transfer_Learned_Samples/" + MODEL_NAME + "/"
MODEL_PATH = "H:\\Transfer_Learned_Models/" + MODEL_NAME + "/"
VISUALIZER = binary_visualizer(64, 3, 3)

if not os.path.exists(SAMPLE_PATH):
    os.makedirs(SAMPLE_PATH)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


def gradient_penalty_loss(y_true, y_pred, averaged_samples):
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


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def load_data():
    #TODO: load minecraft data
    # data = np.load('shapenet_table_combined.npy')

    X = np.load('H:\combined_ingame_binary_augmented_fixed.npy')
    X = X.reshape([-1, 64, 64, 64, 1])
    return X


def sample_images(epoch, latent_dim, generator_model):
    r, c = 3, 3
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator_model.predict(noise)

    gen_imgs[gen_imgs >= .5] = 1
    gen_imgs[gen_imgs < .5] = 0
    gen_imgs.dump(SAMPLE_PATH + '/sample_epoch_' + str(epoch + 1) + '.npy')

    VISUALIZER.draw(gen_imgs, SAMPLE_PATH + '/figures_epoch_' + str(epoch) + '.png')


def train(generator_model, critic_model, gan_model, latent_dim, epochs, batch_size, sample_interval, n_critic):

    # Load the dataset
    X_train = load_data()
    VISUALIZER.draw(X_train[:9], SAMPLE_PATH + '/figures_reals' + '.png')
    valid = -np.ones((batch_size, 1))
    fake = np.ones((batch_size, 1))
    dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
    for epoch in range(epochs):
        for _ in range(n_critic):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            # Sample generator input
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            # Train the critic
            d_loss = critic_model.train_on_batch([imgs, noise], [valid, fake, dummy])

        # ---------------------
        #  Train Generator
        # ---------------------

        g_loss = gan_model.train_on_batch(noise, valid)

        # Plot the progress
        print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0 or epoch == epochs - 1:
            sample_images(epoch, latent_dim, generator_model)
            generator_model.save(MODEL_PATH + "/generator_" + str(epoch), save_traces=True)
            critic_model.save(MODEL_PATH + "/critic_" + str(epoch), save_traces=True)
            gan_model.save(MODEL_PATH + "/GAN_" + str(epoch), save_traces=True)



def build_gan():
    # set parameters for the GAN (these should be the same as the model)
    disable_eager_execution()
    img_x = 64
    img_y = 64
    img_z = 64
    channels = 1
    img_shape = (img_x, img_y, img_z, channels)
    latent_dim = 200

    optimizer = RMSprop(lr=0.0000005)

    # load the pre trained models
    generator = load_model(GENERATOR_PATH)
    critic = load_model(CRITIC_PATH)

    # the models will have input layers attached, so we take the second layer which is the full original model architecture
    gen_model = generator.layers[1]
    critic_model = critic.layers[1]
    critic_model.trainable = True
    print(gen_model.summary())
    print(critic_model.summary())

    # freeze certain layers
    # create a list of layers with trainable parameters
    gen_trainable_layers = []
    critic_trainable_layers = []

    # populate list
    for layer in gen_model.layers:
        if len(layer.trainable_weights):
            gen_trainable_layers.append(layer)
    for layer in critic_model.layers:
        if len(layer.trainable_weights):
            critic_trainable_layers.append(layer)

    # freeze everything but the last two layers in both models
    # TODO: play around with this
    # for layer in gen_trainable_layers[:-3]:
    #     layer.trainable = False
    # for layer in critic_trainable_layers[:2]:
    #     layer.trainable = False



    print(gen_model.summary())
    print(critic_model.summary())


    # rebuild the GAN, basically

    # Image input (real sample)
    real_img = Input(shape=img_shape)

    # Noise input
    z_disc = Input(shape=(latent_dim,))
    # Generate image based of noise (fake sample)
    fake_img = gen_model(z_disc)

    # Discriminator determines validity of the real and fake images
    fake = critic_model(fake_img)
    valid = critic_model(real_img)

    # Construct weighted average between real and fake images
    interpolated_img = RandomWeightedAverage()([real_img, fake_img])
    # Determine validity of weighted sample
    validity_interpolated = critic_model(interpolated_img)

    # Use Python partial to provide loss function with additional
    # 'averaged_samples' argument
    partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=interpolated_img)
    partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

    critic = Model(inputs=[real_img, z_disc], outputs=[valid, fake, validity_interpolated])
    critic.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss], optimizer=optimizer, loss_weights=[1, 1, 10])


    # Sampled noise for input to generator
    z_gen = Input(shape=(latent_dim,))

    # Generate images based of noise
    img = gen_model(z_gen)

    # Discriminator determines validity
    valid = critic_model(img)

    #TODO: figure out if this is maybe messing up our freezing and stuff?
    critic_model.trainable = False

    gan = Model(z_gen, valid)
    gan.compile(loss=wasserstein_loss, optimizer=optimizer)
    print(gan.summary())

    return gen_model, critic, gan

n_critic = 5
generator, critic, gan = build_gan()
train(generator, critic, gan, latent_dim=200, epochs=1500, batch_size=32, sample_interval=199, n_critic=n_critic)

