import numpy as np
import os
import json

from keras.models import load_model

# downsampling?
# seems to work. We get exactly 1/64th (1 / 4x4x4) the number of blocks for each category
# https://stackoverflow.com/questions/62567983/block-reduce-downsample-3d-array-with-mode-function
def blockify(image, block_size):
    shp = image.shape
    out_shp = [s // b for s, b in zip(shp, block_size)]
    reshape_shp = np.c_[out_shp, block_size].ravel()
    nC = np.prod(block_size)
    return image.reshape(reshape_shp).transpose(0, 2, 4, 1, 3, 5).reshape(-1, nC)

def bincount2D_vectorized(a):
    N = a.max() + 1
    a_offs = a + np.arange(a.shape[0])[:, None] * N
    return np.bincount(a_offs.ravel(), minlength=a.shape[0] * N).reshape(-1, N)

def downsample(data):
    downsampled_data = []
    for house in data:
        downsampled_house = bincount2D_vectorized(blockify(house, block_size=(4,4,4))).argmax(1)
        downsampled_house = downsampled_house.reshape([16, 16, 16])
        downsampled_data.append(downsampled_house)
    return np.asarray(downsampled_data)


def sample_and_paint_to_json(generator, painter, num_structs, out_dir):
    # generate structures with generator model
    # gets latent dim from the model
    z_dim = generator.layers[0].input_shape[0][1]

    # generate binary structures
    noise = np.random.normal(0, 1, (num_structs, z_dim))
    structs = generator.predict(noise).squeeze()
    structs[structs >= .5] = 1
    structs[structs < .5] = 0
    structs = structs.astype(int)
    structs = downsample(structs)
    structs = structs.reshape([-1, 16, 16, 16, 1])

    # feed into painter
    painted_houses = painter.predict(structs)
    # convert from one-hot to categorical
    painted_houses = np.argmax(painted_houses, axis=4)

    jset = []
    first = True
    for i, (painted_house, binary_struct) in enumerate(zip(painted_houses, structs)):
        painted_house = np.rot90(painted_house, 1, axes=(1, 2))
        # get house as list
        painted_house = json.dumps(painted_house.tolist())

        binary_struct = np.rot90(binary_struct, 1, axes=(1, 2))
        binary_struct = json.dumps(binary_struct.tolist())

        # write to json file
        painted_id = "painted_" + str(i + 1)
        jdat = {'structure': painted_house}
        jdat['id'] = painted_id
        if first:
            jdat['texture_set'] = ["air", "stonebrick", "dirt", "planks_big_oak", "stone_slab_side"]
            first = False
        jset.append(jdat)
        binary_id = "binary_" + str(i + 1)
        jdat = {'structure': binary_struct}
        jdat['id'] = binary_id
        jset.append(jdat)

    jdump = json.dumps({'structure_set': jset}, indent=3)
    with open(out_dir + "painted.json", 'w+') as f:
        f.write(jdump)


GENERATOR_PATH = 'H:/Transfer_Learned_Models/minecraftGAN_1/generator_2999'
PAINTER_PATH = 'H:/Painter_Models/painter_test8_dropout_fullaugment/painter250'
OUT_PATH = 'H:/Painter_Samples/painter_test8_dropout_fullaugment/gen_structs_test/'

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

NUM_STRUCTS = 10

# load models
generator = load_model(GENERATOR_PATH)
painter = load_model(PAINTER_PATH)

sample_and_paint_to_json(generator, painter, 10, OUT_PATH)
