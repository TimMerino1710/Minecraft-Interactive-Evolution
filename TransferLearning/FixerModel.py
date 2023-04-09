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
import subprocess


import json


class Fixer(Model):
    def __init__(self, name, sub_percent, shape=(64, 64, 64, 1), batches=32, dataset='H:\\joined_cureated_rotated_onehot.npy'):
        super(Fixer, self).__init__()
        self.model_name = name
        self.sample_path = "H:\\Painter_Samples/" + self.model_name + "/"
        self.model_path = "H:\\Painter_Models/" + self.model_name + "/"
        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.batches = batches
        self.shape = shape
        self.test_structs = np.load('H:/gen_houses.npy')
        self.test_structs = self.test_structs[:25]
        self.test_structs = np.expand_dims(self.test_structs, axis=4)

        self.json_out_dirs = []

        # load the painted houses from the dataset
        self.painted_houses = self.load_data(dataset)
        self.painted_houses = np.argmax(self.painted_houses, axis=4)
        self.painted_houses = self.downsample(self.painted_houses)
        self.painted_houses = tf.one_hot(self.painted_houses, 5, dtype=tf.int8).numpy()

        self.binarized_houses = self.binarize_data()
        self.sub_binarized_houses = self.subtractive_noise_augment(percent=sub_percent)

        self.binarized_houses = self.binarized_houses.reshape([-1, 16, 16, 16, 1])
        self.sub_binarized_houses = self.sub_binarized_houses.reshape([-1, 16, 16, 16, 1])

        # downsampling
        # self.painted_houses = np.argmax(self.painted_houses, axis=4)
        # self.painted_houses = self.downsample(self.painted_houses)
        # self.painted_houses = tf.one_hot(self.painted_houses, 5, dtype=tf.int8).numpy()
        print("\n Shape of true data: ", self.binarized_houses.shape, "\n")
        print("\n Shape of train data: ", self.sub_binarized_houses.shape, "\n")


        # split into test and train
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.sub_binarized_houses, self.binarized_houses, test_size=.20)
        self.model = Sequential([
            Input(shape=self.shape),
            # Conv3D(1024,2,activation="relu", padding="same"),
            Conv3D(128, 4, activation="relu", padding="same"),
            Dropout(0.3),
            Conv3D(128, 4, activation="relu", padding="same"),
            Dropout(0.3),
            Conv3D(256, 4, activation="relu", padding="same"),
            Dropout(0.3),
            Conv3D(1, 4, activation="softmax", padding="same"),
            # Dense(self.channels, activation="softmax")
        ])
        self.optimizer = Adam(learning_rate=0.0005)
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        self.model.summary()

    def load_data(self, dataset):
        #minecraft:
        X = np.load(dataset)
        # for binary
        # X = X.reshape([-1, 64, 64, 64, 1])
        return X

    # expects the data in binary downsampled format (16, 16, 16)
    def subtractive_noise_augment(self, percent=.2):
        sub_noise_houses = []
        for house in self.binarized_houses:
            h = np.copy(house)
            # zip into list of (x, y, z) index tuples
            x, y, z = np.nonzero(h)
            rm_idxs = np.random.choice(len(x), int(len(x) * percent), replace=False)
            for idx in rm_idxs:
                h[x[idx], y[idx], z[idx]] = 0
            sub_noise_houses.append(h)
        return np.asarray(sub_noise_houses)


    def random_noise_augment(self, noise_percent=.1):
        for house in self.binarized_houses:
            # sample from a binomial distribution, with a probability noise_percent of each value being a 1
            flip_mask = np.random.choice([0, 1], house.shape, p=[1-noise_percent, noise_percent])
            # todo: figure out how to only flip the bits that are 1 in the flip mask

    def binarize_data(self):
        binarized = np.copy(self.painted_houses)
        # do argmax first
        binarized = np.argmax(binarized, axis=4)
        # convert to binary
        binarized[binarized != 0] = 1
        # binarized = binarized.reshape([-1, 16, 16, 16, 1])
        return binarized

    def train(self, epochs, batch_size, sample_interval=50):
        train_loss = []
        val_loss = []
        ep = 0
        for i in range(epochs // sample_interval):
            history = self.model.fit(x=self.X_train, y=self.Y_train, validation_split=.2, batch_size=batch_size, initial_epoch=ep, epochs=ep + sample_interval, verbose=1)
            ep += sample_interval
            self.evaluate_model(ep)
            self.model.save(self.model_path + "/painter_model_" + str(ep), save_traces=True)
            train_loss += history.history['loss']
            val_loss += history.history['val_loss']

        done = sample_interval * (epochs // sample_interval)
        remaining = epochs - done
        if remaining > 0:
            history = self.model.fit(x=self.X_train, y=self.Y_train, validation_split=.2, batch_size=batch_size, initial_epoch=ep, epochs= ep + remaining, verbose=1)
            self.evaluate_model(ep)
            train_loss += history.history['loss']
            val_loss += history.history['val_loss']

        print("\n Evaluating on test set...\n")
        self.evaluate_model(ep)
        self.render_json_files()
        self.model.save(self.model_path + "/painter" + str(epochs), save_traces=True)

        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.model_path + "loss_graph.png")

    def sample(self, epoch, num_samples=5):
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
            painted_house = np.rot90(painted_house, 1, axes=(1, 2))
            normal_house = np.rot90(normal_house, 1, axes=(1, 2))
            normal_house = json.dumps(normal_house.tolist())
            painted_house = json.dumps(painted_house.tolist())

            # write them to json files
            orig_id = "original_" + str(i+1)
            painted_id = "painted_" + str(i + 1)
            jdat = {'structure': normal_house}
            jdat['id'] = orig_id
            if first:
                jdat['texture_set'] = ["air", "stone_slab_top"]
                first = False
            jset.append(jdat)
            jdat = {'structure': painted_house}
            jdat['id'] = painted_id
            jset.append(jdat)
        jdump = json.dumps({'structure_set': jset}, indent=3)
        with open(sample_epoch_dir + "epoch_" + str(epoch) + ".json", 'w+') as f:
            f.write(jdump)

    def evaluate_model(self, epoch):
        sample_epoch_dir = self.sample_path + "epoch_" + str(epoch) + "/"
        if not os.path.exists(sample_epoch_dir):
            os.makedirs(sample_epoch_dir)

        # get the output of the whole model, just out of curiousity
        painted_test_structs = self.model.predict(self.test_structs)
        painted_test_structs[painted_test_structs >= .5] = 1
        painted_test_structs[painted_test_structs < .5] = 0

        # get the highlighted diff versions between the original generator and the tuned generator
        highlighted_diff_structs = self.get_highlighted_changed_blocks(self.test_structs, painted_test_structs)

        self.gen_render_json([self.test_structs, painted_test_structs, highlighted_diff_structs], ['test_struct', 'fixed', 'fixed_highlighted'], sample_epoch_dir)

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

    # takes a list of structure lists to be rendered, and their names (what the final generated gif will be named)
    def gen_render_json(self, struct_lists, names, out_dir):
        self.json_out_dirs.append(out_dir)
        jset = []
        for structs, name in zip(struct_lists, names):
            structs = self.rot_and_center_structs(structs)
            for i, s in enumerate(structs):
                s = json.dumps(s.tolist())
                id = name + "_" + str(i + 1)
                jdat = {'structure': s}
                jdat['id'] = id

                if len(jset) == 0:
                    jdat['texture_set'] = ["air", "stone_slab_top", "glass_lime", "glass_red"]

                jset.append(jdat)

        jdump = json.dumps({'structure_set': jset}, indent=3)
        with open(out_dir + "eval_structures.json", 'w+') as f:
            f.write(jdump)

    def render_json_files(self):
        for out_dir in self.json_out_dirs:
            file = out_dir + "eval_structures.json"
            subprocess.call(['node', 'mass_render.js', file, 'GIF', out_dir], stdout=subprocess.DEVNULL)

    # replaces blocks that were added by the painter model with a noticable block type
    def get_highlighted_changed_blocks(self, orig_structs, changed_structs):
        highlighted_structs = []
        for i, (orig_struct, changed_struct) in enumerate(zip(orig_structs, changed_structs)):
            highlighted_struct = np.copy(changed_struct)
            # get a boolean mask that is True everywhere the two arrays aren't equal
            different = np.asarray(orig_struct != highlighted_struct)

            # set all the changed blocks in the painted array to green
            highlighted_struct[different] = 2
            different = different.astype(int)

            added = highlighted_struct * different
            missing = orig_struct * different

            # set all the removed blocks in the painted array to red, leaving the greens as added
            missing = missing.astype(bool)
            highlighted_struct[missing] = 3

            highlighted_structs.append(highlighted_struct)

        return np.asarray(highlighted_structs)

    def rot_and_center_structs(self, structs):
        return np.asarray([np.rot90(self.centerHouse(x), 1, axes=[1, 2]) for x in structs])

    def centerHouse(self, house, dim=[16, 16, 16]):
        # find the bounds of the house
        if len(house.shape) == 4:
            house = house.squeeze()
        x, y, z = np.where(house != 0)

        # nothing there?
        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            return house

        xb = [min(x), max(x)]
        yb = [min(y), max(y)]
        zb = [min(z), max(z)]


        # get dimensions of the shape
        xd = xb[1] - xb[0] + 1
        yd = yb[1] - yb[0] + 1
        zd = zb[1] - zb[0] + 1

        # place the house in the middle of the space
        new_house = np.zeros(dim)
        new_house[int((dim[0] - xd) / 2):int((dim[0] + xd) / 2), dim[1] - yd:dim[1], int((dim[2] - zd) / 2):int((dim[2] + zd) / 2)] = house[xb[0]:xb[1] + 1, yb[0]:yb[1] + 1, zb[0]:zb[1] + 1]

        return new_house

    # def exportMod(self, label):
    #     self.model.save(f"painter-{label}.h5")
    #
    # def importMod(self, label):
    #     self.model = keras.models.load_model(f"painter-{label}.h5")

if __name__ == '__main__':
    painter_model = Fixer("fixer_binary2", .30, shape=(16, 16, 16, 1), dataset='H:/joined_cureated_rotated_onehot.npy')
    painter_model.train(epochs=150, batch_size=64, sample_interval=20)

