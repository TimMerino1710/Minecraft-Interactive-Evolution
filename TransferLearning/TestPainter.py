import numpy as np
import os
import json
import subprocess
import shutil

from keras.models import load_model

# downsampling?
# seems to work. We get exactly 1/64th (1 / 4x4x4) the number of blocks for each category
# https://stackoverflow.com/questions/62567983/block-reduce-downsample-3d-array-with-mode-function

# def sample_and_paint_to_json(generator, painter, num_structs, out_dir):
    # generate structures with generator model
    # gets latent dim from the model
    # z_dim = generator.layers[0].input_shape[0][1]
    #
    # # generate binary structures
    # noise = np.random.normal(0, 1, (num_structs, z_dim))
    # structs = generator.predict(noise).squeeze()
    # structs[structs >= .5] = 1
    # structs[structs < .5] = 0
    # structs = structs.astype(int)
    # structs = downsample(structs)
    # structs = structs.reshape([-1, 16, 16, 16, 1])


# replaces blocks that were added by the painter model with a noticable block type
def highlight_added_blocks(binary_houses, painted_houses):
    for binary_house, painted_house in zip(binary_houses, painted_houses):
        # remove extra dimension from binary house
        binary_house = binary_house.squeeze()
        # get boolean array that is 1 wherever the painted model is nonzero
        painted_nonzero = np.asarray(painted_house != 0)
        painted_nonzero = painted_nonzero.astype(int)

        # get a boolean mask that is True everywhere that the painted array has a block and the binary array doesn't
        added = np.asarray(binary_house != painted_nonzero)

        # set all the added blocks in the painted array to a new block type
        painted_house[added] = 5


# replaces blocks that were added by the painter model with a noticable block type
def highlight_added_and_removed_blocks(binary_houses, painted_houses):
    for i, (binary_house, painted_house) in enumerate(zip(binary_houses, painted_houses)):
        # remove extra dimension from binary house
        binary_house = binary_house.squeeze()
        # get boolean array that is 1 wherever the painted model is nonzero
        painted_nonzero = np.asarray(painted_house != 0)
        painted_nonzero = painted_nonzero.astype(int)

        # get a boolean mask that is True everywhere the two arrays aren't equal
        different = np.asarray(binary_house != painted_nonzero)

        # set all the changed blocks in the painted array to green
        painted_house[different] = 5
        different = different.astype(int)

        # different array will be 1 everywhere that there are block differences, and 0 for all blocks that are the same
        # all the blocks that were added by the painter will be 0 in the binary, so multiplying 1 * 0 = 0
        # but all the blocks that were present in binary but not in painter will be 1 * 1 = 1
        added = painted_nonzero * different
        missing = binary_house * different
        # print("House ", i+1, " added: ", np.count_nonzero(added), " removed: ",  np.count_nonzero(missing), " total changed: ", np.count_nonzero(different))


        # set all the removed blocks in the painted array to red, leaving the greens as added
        missing = missing.astype(bool)
        painted_house[missing] = 6




def sample_and_paint_to_json(structs, painter, num_structs, out_dir, painted_filename='painted', highlight_additions=False):
    # get the first num_structs samples from our testing array
    structs = structs[:num_structs]

    # fix dimensions if its not in the shape (num_structs, 16, 16, 16, 1)
    if len(structs.shape) == 4:
        structs = structs.reshape([-1, 16, 16, 16, 1])

    # feed binary into painter
    painted_houses = painter.predict(structs)

    # convert from one-hot to categorical for rendering
    # painted_houses = np.argmax(painted_houses, axis=4)

    # change the painter-added blocks to a different color (cactus block)
    if highlight_additions:
        # highlight_added_blocks(structs, painted_houses)
        highlight_added_and_removed_blocks(structs, painted_houses)


    # write data to file
    jset = []
    first = True
    for i, (painted_house, binary_struct) in enumerate(zip(painted_houses, structs)):
        # rotate the painted and binary houses so they are rendered in the correct orientation
        painted_house = np.rot90(painted_house, 1, axes=(1, 2))
        painted_house = json.dumps(painted_house.tolist())

        binary_struct = np.rot90(binary_struct, 1, axes=(1, 2))
        binary_struct = json.dumps(binary_struct.tolist())

        # write to json file
        painted_id = painted_filename + "_" + str(i + 1)
        jdat = {'structure': painted_house}
        jdat['id'] = painted_id
        if first:
            jdat['texture_set'] = ["air", "stonebrick", "dirt", "planks_big_oak", "stone_slab_side", 'cactus_side', 'bed_feet_top']
            first = False
        jset.append(jdat)
        binary_id = "binary_" + str(i + 1)
        jdat = {'structure': binary_struct}
        jdat['id'] = binary_id
        jset.append(jdat)

    jdump = json.dumps({'structure_set': jset}, indent=3)
    with open(out_dir + "painted.json", 'w+') as f:
        f.write(jdump)

    subprocess.call(['node', 'mass_render.js', out_dir + "painted.json", 'GIF', OUT_PATH], stdout=subprocess.DEVNULL)

#centers a house in the middle of the space
def centerHouse(house, dim=[16,16,16]):
    #find the bounds of the house
    if len(house.shape) == 4:
        house = house.squeeze()
    x,y,z = np.where(house!=0)

    #nothing there?
    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        return house

    xb = [min(x),max(x)]
    yb = [min(y),max(y)]
    zb = [min(z),max(z)]

    # print(xb,yb,zb)

    #get dimensions of the shape
    xd = xb[1]-xb[0]+1
    yd = yb[1]-yb[0]+1
    zd = zb[1]-zb[0]+1

    # print(xd,yd,zd)
    # print("")

    # return house

    #place the house in the middle of the space
    new_house = np.zeros(dim)
    # new_house[0:xd,0:yd,0:zd] = house[xb[0]:xb[1]+1,yb[0]:yb[1]+1,zb[0]:zb[1]+1]
    new_house[int((dim[0]-xd)/2):int((dim[0]+xd)/2),dim[1]-yd:dim[1],int((dim[2]-zd)/2):int((dim[2]+zd)/2)] = house[xb[0]:xb[1]+1,yb[0]:yb[1]+1,zb[0]:zb[1]+1]
    # new_house[int((dim[0]-xd)/2):int((dim[0]+xd)/2),int((dim[1]-yd)/2):int((dim[1]+yd)/2),int((dim[2]-zd)/2):int((dim[2]+zd)/2)] = house[xb[0]:xb[1],yb[0]:yb[1],zb[0]:zb[1]]

    return new_house


def make_comparison_images(structs, non_sub_painter, sub_painter, out_dir, dist=15):
    # get the first num_structs samples from our testing array

    # fix dimensions if its not in the shape (num_structs, 16, 16, 16, 1)
    if len(structs.shape) == 4:
        structs = structs.reshape([-1, 16, 16, 16, 1])

    # feed binary into painter
    painted_houses = sub_painter.predict(structs)
    non_sub_painted_houses = non_sub_painter.predict(structs)

    # convert from one-hot to categorical for rendering
    painted_houses = np.argmax(painted_houses, axis=4)
    non_sub_painted_houses = np.argmax(non_sub_painted_houses, axis=4)

    # change the painter-added blocks to a different color (cactus block)
    highlighted_painted_houses = np.copy(painted_houses)
    highlight_added_and_removed_blocks(structs, highlighted_painted_houses)


    # write data to file
    jset = []
    first = True
    subdirs = []
    print("Writing json...")
    for i, (binary_struct, non_sub_painted_house, sub_painted_house, sub_high_painted_house) in enumerate(zip(structs, non_sub_painted_houses, painted_houses, highlighted_painted_houses)):
        subdir = os.path.join(out_dir, "struct_" + str(i+1))
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        # rotate the painted and binary houses so they are rendered in the correct orientation
        sub_painted_house = np.rot90(sub_painted_house, 1, axes=(1, 2))
        sub_painted_house = centerHouse(sub_painted_house)
        sub_painted_house = json.dumps(sub_painted_house.tolist())

        non_sub_painted_house = np.rot90(non_sub_painted_house, 1, axes=(1, 2))
        non_sub_painted_house = centerHouse(non_sub_painted_house)
        non_sub_painted_house = json.dumps(non_sub_painted_house.tolist())

        sub_high_painted_house = np.rot90(sub_high_painted_house, 1, axes=(1, 2))
        sub_high_painted_house = centerHouse(sub_high_painted_house)
        sub_high_painted_house = json.dumps(sub_high_painted_house.tolist())

        binary_struct = np.rot90(binary_struct, 1, axes=(1, 2))
        binary_struct = centerHouse(binary_struct)
        #TODO: remove
        # make binary more readable by using a special texture
        binary_struct[binary_struct == 1] = 7
        binary_struct = json.dumps(binary_struct.tolist())

        # generate rotated from all angles (0, 90, 180, 270) pngs:
        angle = 0
        for angle in [0, 30, 90, 270, 330]:


            # create IDs that will become file names
            sub_painted_id = "sub_painted_" + str(i + 1) + "_" + str(angle)
            non_sub_painted_id = "non_sub_painted_" + str(i + 1) + "_" + str(angle)
            sub_high_painted_id = "sub_painted_highlighted_" + str(i + 1) + "_" + str(angle)
            binary_id = "binary_" + str(i + 1) + "_" + str(angle)

            jdat = {'structure': binary_struct}
            jdat['id'] = binary_id
            jdat['angle'] = angle
            jdat['distance'] = dist
            jdat['height'] = 4
            if first:
                # jdat['texture_set'] = ["air", "stonebrick", "dirt", "planks_big_oak", "stone_slab_side", 'hardened_clay_stained_lime', 'hardened_clay_stained_red']
                jdat['texture_set'] = ["air", "stonebrick", "dirt", "planks_big_oak", "stone_slab_side", 'glass_lime', 'glass_red', 'stone_slab_top']
                first = False
            jset.append(jdat)

            jdat = {'structure': non_sub_painted_house}
            jdat['id'] = non_sub_painted_id
            jdat['angle'] = angle
            jdat['distance'] = dist
            jdat['height'] = 4
            jset.append(jdat)

            jdat = {'structure': sub_painted_house}
            jdat['id'] = sub_painted_id
            jdat['angle'] = angle
            jdat['distance'] = dist
            jdat['height'] = 4
            jset.append(jdat)

            jdat = {'structure': sub_high_painted_house}
            jdat['id'] = sub_high_painted_id
            jdat['angle'] = angle
            jdat['distance'] = dist
            jdat['height'] = 4
            jset.append(jdat)

            # angle += 30

    jdump = json.dumps({'structure_set': jset}, indent=3)
    with open(out_dir + "paper_painter_figure.json", 'w+') as f:
        f.write(jdump)

    print("Rendering...")
    subprocess.call(['node', 'mass_render.js', out_dir + "paper_painter_figure.json", 'PNG', OUT_PATH], stdout=subprocess.DEVNULL)

    # move files into subfolders
    files = [f for f in os.listdir(out_dir) if f.endswith('.png')]
    for i in range(len(structs)):
        subdir = os.path.join(out_dir, "struct_" + str(i+1))
        search_string = '_' + str(i + 1) + "_"
        belonging_files = [f for f in files if search_string in f]
        new_locs = [os.path.join(subdir, f) for f in belonging_files]
        old_locs = [os.path.join(out_dir, f) for f in belonging_files]
        for (old_loc, new_loc) in zip(old_locs, new_locs):
            shutil.move(old_loc, new_loc)





####################
# autoencoder painter
####################
# PAINTER_PATH = 'H:/Painter_Models/ae_painter_100.h5'
# painter = load_model(PAINTER_PATH, compile=False)
# OUT_PATH = 'H:/Painter_Samples/ae_painter_100_highlighted/'

###################
# standard painter
###################
name = 'painter_model_150'
NON_SUB_PAINTER_PATH = 'H:/Painter_Models/painter_test8_dropout_fullaugment/painter250'
PAINTER_PATH = 'H:/Painter_Models/fixer_binary/' + name
non_sub_painter = load_model(NON_SUB_PAINTER_PATH)
painter = load_model(PAINTER_PATH)
OUT_PATH = 'H:/Painter_Samples/fixer_binary/'

GEN_TEST_STRUCTS_PATH = 'H:/gen_houses.npy'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# load models
# generator = load_model(GENERATOR_PATH)

# load test generated structures
generated_test_samples = np.load(GEN_TEST_STRUCTS_PATH)
NUM_STRUCTS = len(generated_test_samples)
generated_test_samples = generated_test_samples[:25]
# SAMPLE_INDEX = [4,9,11,21,47]
# test_set = generated_test_samples[SAMPLE_INDEX]
#
# make_comparison_images(generated_test_samples, non_sub_painter, painter, OUT_PATH, dist=8)
sample_and_paint_to_json(generated_test_samples, painter, NUM_STRUCTS, OUT_PATH, painted_filename='fixed', highlight_additions=True)
# sample_and_paint_to_json(generated_test_samples, non_sub_painter, NUM_STRUCTS, OUT_PATH, painted_filename='non_sub_painted_highlighted_', highlight_additions=True)
# sample_and_paint_to_json(generated_test_samples, painter, NUM_STRUCTS, OUT_PATH, painted_filename='sub_painted', highlight_additions=False)
# sample_and_paint_to_json(generated_test_samples, painter, NUM_STRUCTS, OUT_PATH, painted_filename='sub_painted_highlighted_', highlight_additions=True)
