import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

HOUSE_NUMPY_DIR = '../house_numpy_files'
HOUSE_OUT_DIR = '../house_combined_numpy_file'

# loads all 3d craft npy files into a list of np arrays
def load_data():
    data = []
    for file in os.listdir(HOUSE_NUMPY_DIR):
        if file.endswith('.npy'):
            data.append(np.load(HOUSE_NUMPY_DIR + '/' + file)[:, :, :, 0])
    return data

# takes a list of np arrays, and calculates the largest size in x, y, z across all entries
def calc_max_axis(data_list):
    max_x = 0
    max_y = 0
    max_z = 0
    for array in data_list:
        shape = array.shape
        x_len, y_len, z_len = shape[0], shape[1], shape[2]
        if x_len > max_x:
            max_x = x_len
        if y_len > max_y:
            max_y = y_len
        if z_len > max_z:
            max_z = z_len
    return max_x, max_y, max_z

# create dataframe with shapes of each
def get_build_dim_df(data_list):
    df = pd.DataFrame(columns = ['x_len', 'y_len', 'z_len'])
    for i, array in enumerate(data_list):
        dim_list = list(array.shape)
        df.loc[i] = dim_list
    df = df.astype(int)
    return df

# returns a a filtered data_list where all samples fit within specified size of (x_max, y_max, z_max)
def cut_to_dim(dim_df, data_list, x_max, y_max, z_max):
    filtered_idx = dim_df.index[(dim_df['x_len'] <= x_max) & (dim_df['y_len']<= y_max) & (dim_df['z_len'] < z_max)].tolist()
    return [data_list[i] for i in filtered_idx]

# write a np array, of size [# samples, xdim, ydim, zdim] to a file for easier loading
def write_to_npy_file(data_list, out_dir=HOUSE_OUT_DIR):
    data_np_array = np.asarray(data_list)
    np.save(out_dir + "/" + "combined.npy", data_np_array)

#TODO: augmentation
# rotate each sample around the x axis in increments of 90 degrees
# should quadruple our number of samples
def rotation_augmentation(data):
    pass

# reflect each sample around the X and/or Y axis
def reflection_augmentation(data):
    pass

# the trickiest augmentation, translate our samples in the X and y dimensions as much as possible
# this will require computing how much we can translate (how much empty space we have on each side, or the buffer)
# decisions about "buffer": do we take the minimum buffer and use that to limit our translation? or do we translate each sample as much as possible within its buffer (will underrepresent larger builds, overrepresent smaller)?
# alternatively, we can increase our maximum size by padding with empty space, that way everything gets equal translations. Will increase dimensionality
# what step size do we take? 1 block? 3? 5?
def translation_augmentation(data):
    pass

data = load_data()
max_x, max_y, max_z = calc_max_axis(data)
print("max x: ", max_x, "    max y: ", max_y, "     max z: ", max_z)

dim_df = get_build_dim_df(data)

filtered_data = cut_to_dim(dim_df, data, 30, 30, 30)
write_to_npy_file(filtered_data)

# dim_df2 = get_build_dim_df(filtered_data)
# dim_df2.hist(bins = 30)
# # dim_df.hist(bins=200)
# plt.show()
