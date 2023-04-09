import numpy as np
import os
import json
import subprocess

from keras.models import load_model

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

# given a start and end vector, generate a list with num_steps vectors that interpolate from start to end, then generate structures using all vectors
def get_interped_structs(generator, start_latent, end_latent, num_steps):
    diff_vec = end_latent - start_latent
    interp_vecs = [start_latent]
    for i in range(num_steps):
        # calculate the "percent" this is between start and end vectors
        perc = (i + 1) / 10.0

        # create interpolation vector by adding the difference vector times the percent to the start vector
        interp_vecs.append(start_latent + np.array(diff_vec * perc))
    interp_vecs.append(end_latent)
    interp_vecs = np.asarray(interp_vecs)
    interp_structs = generator.predict(interp_vecs).squeeze()
    interp_structs[interp_structs >= .5] = 1
    interp_structs[interp_structs < .5] = 0
    interp_structs = interp_structs.astype(int)
    interp_structs = downsample(interp_structs)
    return interp_structs

def mutate(selected_latent_vect, pop_size, sd=.1):
    size = len(selected_latent_vect)
    mutation_vecs = np.random.normal(0, sd, (pop_size - 1, size))
    mutated_latents = [selected_latent_vect]
    for mut_vec in mutation_vecs:
        mutated_latents.append(selected_latent_vect + mut_vec)
    return np.asarray(mutated_latents)

# writes structures to a series of text files (to be used by builder_anim.py)
def make_build_gif(structs, txt_dir, json_file_name, gif_dir):
    txt_file_names = []
    for i, struct in enumerate(structs):
        # rotate the struct to correct orientation, then dump to string
        binary_struct = np.rot90(struct, 1, axes=(1, 2))
        # binary_struct = json.dumps(binary_struct.tolist())

        txt_file_name = txt_dir + 'struct_' + str(i) + '.txt'
        json.dump(binary_struct.tolist(), open(txt_file_name, 'w+'))

        txt_file_names.append(txt_file_name)

    # call builder_anim
    subprocess.call(['python', 'builder_anim.py'] + txt_file_names + ['--filename', json_file_name, '--rotate', '--cycle_frames', '60', '--distance', '20', '--struct_delay', '200', '--block_delay', '0'])
    # subprocess.call(['python', 'builder_anim.py'] + txt_file_names + ['--filename', json_file_name, '--struct_delay', '0', '--block_delay', '0'])
    # call build_js to generate interpolation gif
    subprocess.call(['node', 'builder.js', json_file_name, 'GIF', gif_dir])


def generate_starting_pop(generator, num_structs):
    # gets latent dim from the model
    z_dim = generator.layers[0].input_shape[0][1]
    # generate binary structures
    noise = np.random.normal(0, 1, (num_structs, z_dim))
    structs = generator.predict(noise).squeeze()
    structs[structs >= .5] = 1
    structs[structs < .5] = 0
    structs = structs.astype(int)
    structs = downsample(structs)
    return noise, structs


def generate_pop(generator, latents):
    structs = generator.predict(latents).squeeze()
    structs[structs >= .5] = 1
    structs[structs < .5] = 0
    structs = structs.astype(int)
    structs = downsample(structs)
    return structs


def paint_structs(painter, structs):
    # fix dimensions if its not in the shape (num_structs, 16, 16, 16, 1)
    if len(structs.shape) == 4:
        structs = structs.reshape([-1, 16, 16, 16, 1])
    # feed binary into painter
    painted_houses = painter.predict(structs)
    painted_houses = np.argmax(painted_houses, axis=4)
    return painted_houses


# given a start and end vector, generate a list with num_steps vectors that interpolate from start to end, then generate structures using all vectors
def get_interped_structs(generator, start_latent, end_latent, num_steps):
    diff_vec = end_latent - start_latent
    interp_vecs = [start_latent]
    for i in range(num_steps):
        # calculate the "percent" this is between start and end vectors
        perc = (i + 1) / 10.0

        # create interpolation vector by adding the difference vector times the percent to the start vector
        interp_vecs.append(start_latent + np.array(diff_vec * perc))
    interp_vecs.append(end_latent)
    interp_vecs = np.asarray(interp_vecs)
    interp_structs = generator.predict(interp_vecs).squeeze()
    interp_structs[interp_structs >= .5] = 1
    interp_structs[interp_structs < .5] = 0
    interp_structs = interp_structs.astype(int)
    interp_structs = downsample(interp_structs)
    return interp_structs


def render_samples(structs, out_dir):
    # write data to file
    jset = []
    first = True
    for i, struct in enumerate(structs):
        # rotate thebinary houses so they are rendered in the correct orientation
        binary_struct = np.rot90(struct, 1, axes=(1, 2))
        binary_struct = json.dumps(binary_struct.tolist())

        # write to json file
        id = "pop_" + str(i + 1)
        jdat = {'structure': binary_struct}
        jdat['id'] = id
        if first:
            jdat['texture_set'] = ["air", "stonebrick", "dirt", "planks_big_oak", "stone_slab_side", 'cactus_side']
            first = False
        jset.append(jdat)

    jdump = json.dumps({'structure_set': jset}, indent=3)
    with open(out_dir + "gen_structs.json", 'w+') as f:
        f.write(jdump)

    subprocess.call(['node', 'mass_render.js', out_dir + "gen_structs.json", 'GIF', out_dir], stdout=subprocess.DEVNULL)


POP_SIZE = 10
ITER = 10

GENERATOR_PATH = 'H:/generator_2999'
PAINTER_PATH = 'H:/Painter_Models/painter_30perc/painter500'
BASE_PATH = 'H:/MutationExperiments/'
generator = load_model(GENERATOR_PATH)
painter = load_model(PAINTER_PATH)

for i in range(3):
    EXPERIMENT_NAME = '60tenths_sd' + str(i + 1)
    path = BASE_PATH + EXPERIMENT_NAME + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    family_tree = []

    # starting population
    # latent_vecs, structs = generate_starting_pop(generator, POP_SIZE)
    # painted_structs = paint_structs(painter, structs)
    # render_samples(painted_structs, path)

    latent_vecs = np.load('H:/MutationExperiments/starting_latent_vectors.npy')
    structs = generate_pop(generator, latent_vecs)
    painted_structs = paint_structs(painter, structs)
    render_samples(painted_structs, path)
    # np.save('H:/MutationExperiments/starting_latent_vectors.npy', latent_vecs)
    # simulated IE loop
    selections = 0
    for i in range(ITER):
        # create a nested dir
        sel_idx = 1
        path = path + "/" + str(sel_idx + 1) + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        selections += 1
        # mutate based on selected vector
        selected_vec = latent_vecs[sel_idx]
        family_tree.append(painted_structs[sel_idx])
        latent_vecs = mutate(selected_vec, POP_SIZE, sd=.60)
        structs = generate_pop(generator, latent_vecs)
        painted_structs = paint_structs(painter, structs)
        render_samples(painted_structs, path)

    family_tree.append(painted_structs[1])
    print("Len family tree: ", len(family_tree))
    print("Number of selections made: ", selections)
    path = BASE_PATH + EXPERIMENT_NAME + "/"
    path = path + '/family_tree/'
    if not os.path.exists(path):
        os.makedirs(path)
    make_build_gif(family_tree, path, "family_tree.json", path)




