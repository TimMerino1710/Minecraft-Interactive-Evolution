import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from nbtschematic import SchematicFile

HOUSE_NUMPY_DIR = '../house_numpy_files'
HOUSE_OUT_DIR = '../house_combined_numpy_file'
SANITY_CHECK_DIR = 'sanity_checks'

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
def write_to_npy_file(data_list, file_name, out_dir=HOUSE_OUT_DIR):
    data_np_array = np.asarray(data_list)
    np.save(out_dir + "/" + file_name, data_np_array)

# pads all arrays in data_list to be size (max_x, max_y, max_z) by padding edges with 0s
def pad_arrays(data_list, max_x, max_y, max_z):
    padded_data_list = []
    for array in data_list:
        shape = array.shape
        cur_x, cur_y, cur_z = shape[0], shape[1], shape[2]
        padded = np.pad(array, ((0, max_x - cur_x), (0, max_y - cur_y), (0, max_z - cur_z)), mode='constant')
        padded_data_list.append(padded)
    return padded_data_list

# turns every block into stone, for testing with binary voxel models
def convert_to_stone_only(data_list):
    # for array in data_list:
    for array in data_list:
        array[np.where(array != 0)] = 1
    return data_list

# convert npy files in dir to schematic files
def convert_to_schem(dir):
    count = 1
    for dirname, _, filelist in os.walk(dir):
        for file in filelist:
            if file.endswith('.npy'):
                print(file)
                data = np.load(dirname + "/" + file)
                arr_axes = (data.shape[0], data.shape[1], data.shape[2])
                sf = SchematicFile(shape=arr_axes)
                assert sf.blocks.shape == arr_axes
                for index, block_id in np.ndenumerate(data):
                    sf.blocks[index[0], index[1], index[2]] = block_id
                sf.save(dir + "/" + file + ".schematic")
                count += 1

# debug method to output an original array as a npy file, as well as a list of transformed arrays, to confirm that any augmentations / manipulations are working correctly
def sanity_check(array, others):
    np.save(SANITY_CHECK_DIR + "/before.npy", array)
    for i, array in enumerate(others):
        np.save(SANITY_CHECK_DIR + "/after" + str(i) + ".npy", array)


# from paper:
# "We change the binary voxel range from {0,1} to {-1,5} to encourage the network to pay more attention to positive entries"
def scale_data(data):
    scaled_data = []
    for array in data:
        new = array.astype(int)
        new[new == 0] = -1
        new[new == 1] = 5
        scaled_data.append(new)
    return scaled_data

# rotates samples around an axis (the Z axis? not sure, minecraft axes are confusing, but it rotates in the way you would expect)
def rotation_augmentation(data):
    rotated_data = []
    for array in data:
        rotated_90 = np.rot90(array, axes=(1, 2))
        rotated_180 = np.rot90(rotated_90, axes=(1, 2))
        rotated_270 = np.rot90(rotated_180, axes=(1, 2))
        rotated_data.append(array)
        rotated_data.append(rotated_90)
        rotated_data.append(rotated_180)
        rotated_data.append(rotated_270)
    return rotated_data

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
# max_x, max_y, max_z = calc_max_axis(data)
# print("max x: ", max_x, "    max y: ", max_y, "     max z: ", max_z)

dim_df = get_build_dim_df(data)

filtered_data = cut_to_dim(dim_df, data, 32, 32, 32)
padded_filtered_data = pad_arrays(filtered_data, 32, 32, 32)
stone_only = convert_to_stone_only(padded_filtered_data)
scaled = scale_data(stone_only)

# sanity_check(stone_only[0], [scaled[0]])
# convert_to_schem(SANITY_CHECK_DIR)
rotated = rotation_augmentation(scaled)

# sanity_check(rotated[0], rotated[1:])
# convert_to_schem(SANITY_CHECK_DIR)
# sanity_check(filtered_data[25], padded_filtered_data[25])
write_to_npy_file(rotated, "stoneonly_combined_rotated_scaled.npy")

# dim_df2 = get_build_dim_df(filtered_data)
# dim_df2.hist(bins = 30)
# # dim_df.hist(bins=200)
# plt.show()
