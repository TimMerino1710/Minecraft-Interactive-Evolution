import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class binary_visualizer():
    def __init__(self, dim, rows, cols):
        self.dimension = dim
        self.rows = rows
        self.cols = cols

    def draw(self, combined_gen_samples, out_loc):
        fig = plt.figure()
        # ax = fig.add_subplot(4, 3, 1, projection='3d')
        combined_gen_samples = np.squeeze(combined_gen_samples)
        for i in range(self.rows * self.cols):
            # define subplot
            # ax = fig.add_subplot( projection='3d')
            ax = fig.add_subplot(self.rows, self.cols, i + 1, projection='3d')
            # turn off axis
            # plt.axis('off')
            # plot raw pixel data
            # ax.set_aspect('equal')
            # print(combined_gen_samples.shape)
            # print(combined_gen_samples[i].shape)
            ax.voxels(combined_gen_samples[i], edgecolor="k", facecolors='red', linewidth=0.5)

            # plt.imshow(X[i, :, :, 0], cmap='gray_r')
        # save plot to file
        plt.savefig(out_loc)
        plt.close(fig)


class categorical_visualizer():
    def __init__(self, dim, rows, cols):
        self.dimension = dim
        self.rows = rows
        self.cols = cols

    def draw(self, combined_gen_samples, out_loc):
        fig = plt.figure()
        combined_gen_samples = np.squeeze(combined_gen_samples)
        combined_gen_samples[combined_gen_samples != 0] = 1
        for i in range(self.rows * self.cols):

            ax = fig.add_subplot(self.rows, self.cols, i + 1, projection='3d')
            ax.voxels(combined_gen_samples[i], edgecolor="k", facecolors='red', linewidth=0.5)

        plt.savefig(out_loc)
        plt.close(fig)

def draw_struct(array):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.voxels(array, edgecolor="k")
    plt.show()

def draw_rounded_structs(orig, copies):
    fig = plt.figure()
    ax = fig.add_subplot(4, 3, 1, projection='3d')
    ax.set_aspect('equal')
    ax.voxels(orig, edgecolor="k")

    for i, array in enumerate(copies):
        ax = fig.add_subplot(4, 3, i + 2, projection='3d')
        ax.set_aspect('equal')
        ax.voxels(array, edgecolor="k")
    plt.show()

def draw_generated_rounded_structs(orig, copies):
    fig = plt.figure()
    ax = fig.add_subplot(4, 3, 1, projection='3d')
    ax.set_aspect('equal')
    ax.voxels(orig, edgecolor="k")

    for i, array in enumerate(copies):
        ax = fig.add_subplot(4, 3, i + 2, projection='3d')
        ax.set_aspect('equal')
        ax.voxels(array, edgecolor="k")
    plt.show()

def round_arrays(arrays, thresh):
    rounded = []
    for array in arrays:
        copy = np.copy(array)
        copy[np.where(copy >= thresh)] = 1
        copy[np.where(copy < thresh)] = 0
        rounded.append(copy)
    return rounded

def get_generated(dir):
    generated_data = []
    for dirname, subdirlist, filelist in os.walk(dir):
        for file in filelist:
            if file.endswith('.npy'):
                data = load_array(dirname + file)
                generated_data.append(data)
    return generated_data

def write_rounded(rounded_arrays):
    for i, array in enumerate(rounded_arrays):
        np.save('generated_samples/rounded/rounded_' + str(i) + ".npy", array)

def load_array(array_path):
    return np.load(array_path).squeeze()

def get_rounded_arrays(non_binary_array):
    rounded_arrays = []

    i = -1
    while i <= 1:
        copy = np.copy(non_binary_array)
        copy[np.where(copy >= i)] = 1
        copy[np.where(copy < i)] = 0
        rounded_arrays.append(copy)
        i += .2
    return rounded_arrays



# generated_data = get_generated("generated_samples/")
# rounded_arrays = round_arrays(generated_data, -1)
# write_rounded(rounded_arrays)
# orig_array = load_array('decoded_test_data/test_1.npy')
# recon_array = load_array('decoded_test_data/test_reconstructed_1.npy')
# orig_array = load_array('decoded_test_data/test_19.npy')
# recon_array = load_array('generated_samples/generated1500_2.npy')
# print(np.unique(recon_array))
# copies = get_rounded_arrays(recon_array)
# draw_rounded_structs(orig_array, copies)