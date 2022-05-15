import tensorflow as tf
import numpy as np
from nbtschematic import SchematicFile
import os

compression_list = [[0, 6, 26, 27, 28, 30, 55, 63, 65, 66, 68, 69, 70, 72, 77, 97, 104, 117, 127, 131, 132, 143, 144, 147, 148, 149, 167, 171, 50, 51, 76, 105, 123],[1, 4, 7, 29, 33, 34, 46, 48, 49, 52, 54, 61, 87, 89, 98, 45, 112, 120, 121, 139, 155, 158, 168, 169, 201, 202, 206, 215, 216, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 35, 14, 15, 16, 21, 56, 73, 129, 153],[2, 3, 82, 88, 159, 172, 173, 214],[5, 17, 25, 47, 58, 84, 96, 116, 130, 140, 146, 151, 154, 162],[8, 9, 10, 11, 213],[12, 13, 19, 24, 179],[18, 31, 32, 81, 86, 91, 103, 106, 161, 170, 199, 200, 207],[20, 92, 102, 160, 95],[22, 23, 41, 42, 57, 118, 133, 138, 145, 152, 165],[37, 38, 39, 40, 59, 83, 110, 115, 141, 142, 175],[43, 44, 92, 125, 126, 181, 182, 204, 205],[53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164, 180, 203],[64, 71, 193, 194, 195, 196, 197],[78, 79, 80, 174],[85, 101, 107, 113, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 198],]

def decode_gan_samples(file):
    data = np.load(file, allow_pickle=True)
    data = tf.argmax(data, axis=4).numpy()
    return data

def decompress(data):
    for i in range(len(compression_list)):
        data[data == i] = compression_list[i][0]
    return data

def convert_to_schem(data, out_dir):
    # print(len(data))
    for i in range(len(data)):

        build = data[i]
        # print(build.shape)
        arr_axes = (build.shape[0], build.shape[1], build.shape[2])
        sf = SchematicFile(shape=arr_axes)
        assert sf.blocks.shape == arr_axes
        for index, block_id in np.ndenumerate(build):
            sf.blocks[index[0], index[1], index[2]] = block_id
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        sf.save(out_dir + "/generated_" + str(i) + ".schematic")

def process_binary_samples(sample_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for file in os.listdir(sample_dir):
        sample = np.load(sample_dir + file)
        # print(sample.shape)
        file_name = os.path.splitext(file)[0]
        arr_axes = (sample.shape[0], sample.shape[1], sample.shape[2])
        sf = SchematicFile(shape=arr_axes)
        assert sf.blocks.shape == arr_axes
        for index, block_id in np.ndenumerate(sample):
            sf.blocks[index[0], index[1], index[2]] = block_id
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        sf.save(out_dir + "/" + file_name + ".schematic")

def process_binary_samples_combined(sample_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for file in os.listdir(sample_dir):
        combined_matrix = np.load(sample_dir + file, allow_pickle=True)
        combined_matrix[combined_matrix < .5] = 0
        combined_matrix[combined_matrix >= .5] = 1
        # print(combined_matrix.shape)
        file_name = os.path.splitext(file)[0]
        for i in range(len(combined_matrix)):


            sample = combined_matrix[i]
            uniques, counts = np.unique(sample, return_counts=True)
            # print(dict(zip(np.array(uniques), counts)))
            # print(sample.shape)
            arr_axes = (sample.shape[0], sample.shape[1], sample.shape[2])
            sf = SchematicFile(shape=arr_axes)
            assert sf.blocks.shape == arr_axes
            for index, block_id in np.ndenumerate(sample):
                sf.blocks[index[0], index[1], index[2]] = block_id
            sf.save(out_dir + "/" + file_name + "_" + str(i) + ".schematic")

def process_vae_samples(sample_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # loop over every output file
    for file in os.listdir(sample_dir):
        if file.endswith('.npy'):
            file_name = os.path.splitext(file)[0]

            # load data and then turn back into mincraft block indexes by inverting our compression list
            combined_samples = np.load(sample_dir + file, allow_pickle=True)
            combined_samples = decompress(combined_samples)
            print(file_name, ", shape: ", combined_samples.shape)

            # write each sample to file
            for i in range(len(combined_samples)):

                sample = combined_samples[i]
                uniques, counts = np.unique(sample, return_counts=True)
                print("uniques: ", uniques, ", counts: ", counts)
                arr_axes = (sample.shape[0], sample.shape[1], sample.shape[2])
                sf = SchematicFile(shape=arr_axes)
                assert sf.blocks.shape == arr_axes
                for index, block_id in np.ndenumerate(sample):
                    sf.blocks[index[0], index[1], index[2]] = block_id
                sf.save(out_dir + "/" + file_name + "_" + str(i) + ".schematic")

# =========== One-hot encoded code ===========
# categorical = decode_gan_samples('20_stone_only/generated_samples/samples_epoch100.npy')
# decompressed = decompress(categorical)
# uniques, counts = np.unique(decompressed, return_counts=True)
# print(print(dict(zip(np.array(uniques), counts))))
# convert_to_schem(decompressed, '20_stone_only/generated_samples/samples_epoch100')

process_vae_samples('GAN_generated_samples/WGANGP_20_deep_categorical_3/test/', 'GAN_generated_samples/WGANGP_20_deep_categorical_3/test/schematics')

# ============= binary code ===========
# process_binary_samples_combined('GAN_generated_samples/WGANGP_binary_20/', 'GAN_generated_samples/WGANGP_binary_20/schematics')
# process_binary_samples('GAN_generated_samples/binaryGAN_biggerkernels_higherDLR/', 'GAN_generated_samples/binaryGAN_biggerkernels_higherDLR/schematics')