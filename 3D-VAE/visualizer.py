import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def draw_struct(array):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.voxels(array, edgecolor="k")
    plt.show()

def draw_rounded_structs(orig, copies):
    fig = plt.figure()
    ax = fig.add_subplot(5, 5, 1, projection='3d')
    ax.set_aspect('equal')
    ax.voxels(orig, edgecolor="k")

    for i, array in enumerate(copies):
        ax = fig.add_subplot(5, 5, i + 2, projection='3d')
        ax.set_aspect('equal')
        ax.voxels(array, edgecolor="k")
    plt.show()



def load_array(array_path):
    return np.load(array_path).squeeze()

def get_rounded_arrays(non_binary_array):
    rounded_arrays = []

    i = -1
    while i <= 2:
        copy = np.copy(non_binary_array)
        copy[np.where(copy >= i)] = 1
        copy[np.where(copy < i)] = 0
        rounded_arrays.append(copy)
        i += .2
    return rounded_arrays



orig_array = load_array('decoded_test_data/test_3.npy')
recon_array = load_array('decoded_test_data/test_reconstructed_3.npy')
copies = get_rounded_arrays(recon_array)
draw_rounded_structs(orig_array, copies)