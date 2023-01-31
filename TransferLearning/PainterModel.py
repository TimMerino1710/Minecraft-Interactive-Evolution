import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

from keras.models import Model
from keras import backend as K
from keras import Input
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Conv3DTranspose, LeakyReLU, Reshape, ReLU, Conv3D, Flatten, MaxPooling3D, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import json


class Painter(Model):
    def __init__(self, shape=(64, 64, 64, 1), batches=32):
        super(Painter, self).__init__()
        self.model_name = 'painter_test8_dropout_fullaugment'
        self.sample_path = "H:\\Painter_Samples/" + self.model_name + "/"
        self.model_path = "H:\\Painter_Models/" + self.model_name + "/"
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.batches = batches
        self.shape = shape
        # self.channels = channels
        self.painted_houses = self.load_data()

        # downsampling
        self.painted_houses = np.argmax(self.painted_houses, axis=4)
        self.painted_houses = self.downsample(self.painted_houses)
        self.painted_houses = tf.one_hot(self.painted_houses, 5, dtype=tf.int8).numpy()
        print("\n Shape of loaded data: ", self.painted_houses.shape, "\n")


        self.binarized_houses = self.binarize_data()
        print("\n Shape of binarized data: ", self.binarized_houses.shape, "\n")
        self.channels = self.painted_houses.shape[-1]
        print("\n No. Channels: : ", self.channels, "\n")

        # split into test and train
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.binarized_houses, self.painted_houses, test_size=.20)
        self.model = Sequential([
            Input(shape=self.shape),
            # Conv3D(1024,2,activation="relu", padding="same"),
            Conv3D(256, 4, activation="relu", padding="same"),
            Dropout(0.3),
            Conv3D(128, 4, activation="relu", padding="same"),
            Dropout(0.3),
            Conv3D(self.channels, 4, activation="softmax", padding="same"),
            # Dense(self.channels, activation="softmax")
        ])
        self.optimizer = Adam(learning_rate=0.0005)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
        self.model.summary()

    def load_data(self):
        #minecraft:
        X = np.load('H:\\joined_cureated_stretched_rotated_categorical_onehot.npy')
        # for binary
        # X = X.reshape([-1, 64, 64, 64, 1])

        return X

    def binarize_data(self):
        binarized = np.copy(self.painted_houses)
        # do argmax first
        binarized = np.argmax(binarized, axis=4)
        # convert to binary
        binarized[binarized != 0] = 1
        binarized = binarized.reshape([-1, 16, 16, 16, 1])
        return binarized

    def train(self, epochs, batch_size, sample_interval=50):
        # self.sample(0, 3)
        ep = 0
        # hacky way to sample at checkpoints?
        for i in range(epochs // sample_interval):
            self.model.fit(x=self.X_train, y=self.Y_train, validation_split=.2, batch_size=batch_size, initial_epoch=ep, epochs=ep + sample_interval, verbose=1)
            ep += sample_interval
            self.sample(ep, 3)

        done = sample_interval * (epochs // sample_interval)
        remaining = epochs - done
        if remaining > 0:
            self.model.fit(x=self.X_train, y=self.Y_train, validation_split=.2, batch_size=batch_size, initial_epoch=ep, epochs= ep + remaining, verbose=1)
            self.sample(epochs, 3)

        print("\n Evaluating on test set...\n")
        self.model.evaluate(self.X_test, self.Y_test, verbose=1)
        self.model.save(self.model_path + "/painter" + str(epochs), save_traces=True)


    def sample(self, epoch, num_samples=5):
        print("\n Sampling on epoch", epoch, "...")
        sample_epoch_dir = self.sample_path + "epoch_" + str(epoch) + "/"
        if not os.path.exists(sample_epoch_dir):
            os.makedirs(sample_epoch_dir)

        # randomly sample pairs of binary and regular houses
        random_idxs = random.sample(range(0, len(self.X_test)), num_samples)
        bin_houses = self.X_test[random_idxs]
        normal_houses = self.Y_test[random_idxs]

        # get painted version of the binary houses
        painted_houses = self.model.predict(bin_houses)

        # convert from one-hot to categorical
        normal_houses = np.argmax(normal_houses, axis=4)
        painted_houses = np.argmax(painted_houses, axis=4)

        # write to file
        jset = []
        first = True
        for i, (normal_house, painted_house) in enumerate(zip(normal_houses, painted_houses)):
            # get these as lists
            normal_house = json.dumps(normal_house.tolist())
            painted_house = json.dumps(painted_house.tolist())

            # write them to json files
            orig_id = "original_" + str(i+1)
            painted_id = "painted_" + str(i + 1)
            jdat = {'structure': normal_house}
            jdat['id'] = orig_id
            if first:
                jdat['texture_set'] = ["air", "stonebrick", "dirt", "planks_big_oak", "stone_slab_side"]
                first = False
            jset.append(jdat)
            jdat = {'structure': painted_house}
            jdat['id'] = painted_id
            jset.append(jdat)
            # jset = []
            # jdat = {'structure': normal_house}
            # jdat['id'] = file_name
            # jdat['texture_set'] = ["air", "stonebrick", "dirt", "planks_big_oak", "stone_slab_side"]
            # jset.append(jdat)
            # jdump = json.dumps({'structure_set': jset}, indent=3)
            # with open(sample_epoch_dir + file_name, 'w+') as f:
            #     f.write(jdump)
            #
            # file_name = "painted_" + str(i + 1) + '.json'
            # jset = []
            # # painted_house = str(painted_house)
            # # painted_house = [h.strip() for h in painted_house]
            # jdat = {'structure': painted_house}
            # jdat['id'] = file_name
            # jdat['texture_set'] = ["air", "stonebrick", "dirt", "planks_big_oak", "stone_slab_side"]
            # jset.append(jdat)
            # jdump = json.dumps({'structure_set': jset}, indent=3)
            # with open(sample_epoch_dir + file_name, 'w+') as f:
            #     f.write(jdump)
        jdump = json.dumps({'structure_set': jset}, indent=3)
        with open(sample_epoch_dir + "epoch_" + str(epoch) + ".json", 'w+') as f:
            f.write(jdump)


    # downsampling?
    # seems to work. We get exactly 1/64th (1 / 4x4x4) the number of blocks for each category
    # https://stackoverflow.com/questions/62567983/block-reduce-downsample-3d-array-with-mode-function
    def blockify(self, image, block_size):
        shp = image.shape
        out_shp = [s // b for s, b in zip(shp, block_size)]
        reshape_shp = np.c_[out_shp, block_size].ravel()
        nC = np.prod(block_size)
        return image.reshape(reshape_shp).transpose(0, 2, 4, 1, 3, 5).reshape(-1, nC)

    def bincount2D_vectorized(self, a):
        N = a.max() + 1
        a_offs = a + np.arange(a.shape[0])[:, None] * N
        return np.bincount(a_offs.ravel(), minlength=a.shape[0] * N).reshape(-1, N)

    def downsample(self, data):
        downsampled_data = []
        for house in data:
            downsampled_house = self.bincount2D_vectorized(self.blockify(house, block_size=(4,4,4))).argmax(1)
            downsampled_house = downsampled_house.reshape([16, 16, 16])
            downsampled_data.append(downsampled_house)
        return np.asarray(downsampled_data)

    # def exportMod(self, label):
    #     self.model.save(f"painter-{label}.h5")
    #
    # def importMod(self, label):
    #     self.model = keras.models.load_model(f"painter-{label}.h5")

if __name__ == '__main__':
    painter_model = Painter(shape=(16, 16, 16, 1))
    painter_model.train(epochs=250, batch_size=64, sample_interval=20)
