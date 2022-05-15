from bz2 import compress
from keras.models import load_model
import keras.backend as K
import numpy as np
import random
import grpc
import minecraft_pb2_grpc
from minecraft_pb2 import *
import msvcrt
import sys
import argparse
from fitness_functions import getch

channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

compression_list = [[0, 6, 26, 27, 28, 30, 55, 63, 65, 66, 68, 69, 70, 72, 77, 97, 104, 117, 127, 131, 132, 143, 144, 147, 148, 149, 167, 171, 50, 51, 76, 105, 123],[1, 4, 7, 29, 33, 34, 46, 48, 49, 52, 54, 61, 87, 89, 98, 45, 112, 120, 121, 139, 155, 158, 168, 169, 201, 202, 206, 215, 216, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 35, 14, 15, 16, 21, 56, 73, 129, 153],[2, 3, 82, 88, 159, 172, 173, 214],[5, 17, 25, 47, 58, 84, 96, 116, 130, 140, 146, 151, 154, 162],[8, 9, 10, 11, 213],[12, 13, 19, 24, 179],[18, 31, 32, 81, 86, 91, 103, 106, 161, 170, 199, 200, 207],[20, 92, 102, 160, 95],[22, 23, 41, 42, 57, 118, 133, 138, 145, 152, 165],[37, 38, 39, 40, 59, 83, 110, 115, 141, 142, 175],[43, 44, 92, 125, 126, 181, 182, 204, 205],[53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164, 180, 203],[64, 71, 193, 194, 195, 196, 197],[78, 79, 80, 174],[85, 101, 107, 113, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 198],]
# compression_list = [[0, 6, 26, 27, 28, 30, 55, 63, 65, 66, 68, 69, 70, 72, 77, 97, 104, 117, 127, 131, 132, 143, 144, 147, 148, 149, 167, 171, 50, 51, 76, 105, 123, 64, 71, 193, 194, 195, 196, 197, 8, 9, 10, 11, 213],[1, 4, 7, 29, 33, 34, 46, 48, 49, 52, 54, 61, 87, 89, 98, 45, 112, 120, 121, 139, 155, 158, 168, 169, 201, 202, 206, 215, 216, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 35, 14, 15, 16, 21, 56, 73, 129, 153],[2, 3, 82, 88, 159, 172, 173, 214, 12, 13, 19, 24, 179, 78, 79, 80, 174],[5, 17, 25, 47, 58, 84, 96, 116, 130, 140, 146, 151, 154, 162],[18, 31, 32, 81, 86, 91, 103, 106, 161, 170, 199, 200, 207],[20, 92, 102, 160, 95],[22, 23, 41, 42, 57, 118, 133, 138, 145, 152, 165],[37, 38, 39, 40, 59, 83, 110, 115, 141, 142, 175],[43, 44, 92, 125, 126, 181, 182, 204, 205],[53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164, 180, 203],[85, 101, 107, 113, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 198]]

BOUNDS_WORLD = [[-30000000, 29999999],  [4, 255], [-30000000, 29999999]]

DIRECTIONS = ["N", "W", "S", "E", "U", "D"]
DIRECTIONS_2D = ["W", "E", "U", "D"]
ORIENTATIONS = ["O.N", "O.W", "O.S", "O.E", "O.U", "O.D"]


gan_model_path = '../3D-VAE/GAN_models/WGANGP_16_smaller_compression_1/generator_200.h5'
vae_model_path = '../3D-VAE/VAE_models/16_3DVAE_newlossy/16_3DVAE_newlossy/decoder.h5'

pop_size = 5

mutation_prob = .05
mutation_amt = .1

# # TODO: get the size of the z dimension by reading the first layer of the model
# z_dim = 200

# generates num_pop number of random normal vectors with size z_dim
def create_starting_population_gan(num_pop, z_dim):
    return np.random.normal(0, 1, (num_pop, z_dim))

# turns integer values back into block IDs by inverting our compression
def decompress(data):
    for i in range(len(compression_list)):
        data[data == i] = compression_list[i][0]
    return data

# take latent vectors, and generate structures from them
def generate_from_latent(model, latent_vectors):
    generated_structures = model.predict(latent_vectors)

    # a categorical GAN will output one-hot encoded, so this must turned back into categorical
    generated_structures = np.argmax(generated_structures, axis=4)

    # this results in integer values between 0 and 14. These must be decompressed back into their actual minecraft blockids to render properly
    generated_structures = decompress(generated_structures)

    return generated_structures

"""RENDERING HELPERS"""

def bound_coordinate(value, coord):
    """
    Restrict the coordinate to the bounds.
    INPUT:
        value: a value
        coord: the index of the coordinate (0,1,2)

    OUTPUT:
        the value bounded according the bounds registered in BOUNDS_MINECRAFT above.
    """
    low = BOUNDS_WORLD[coord][0]
    high = BOUNDS_WORLD[coord][1]
    return max(low, min(high, value))

def bounded(position):
    """
    Bounds the position according to BOUNDS_WORLD.
    INPUT:
        position: a 3D position.

    OUTPUT:
        bounded_position: a 3D position, within the boundaries given by BOUNDS_WORLD

    """
    bounded_position = [bound_coordinate(position[0], 0), bound_coordinate(
        position[1], 1), bound_coordinate(position[2], 2)]
    return bounded_position

def build_zone(blocks, offset):
    """
    Build a 3D structure, given by a tensor specifiying the value of each block type at each position (3D), and possibly orientations
    Inputs:
        blocks: np array size Mx*My*MZ, where Mx,My,Mz are bounds given as input.

    """
    positions = []
    blocks_index = []

    for x in range(blocks.shape[0]):
        for y in range(blocks.shape[1]):  # this is height in minecraft
            for z in range(blocks.shape[2]):
                index = int(blocks[x, y, z])
                # if not index == -1:  # AS INDEX - 1 means air block
                #     try:
                #         blocks_index.append(compression_list[0][index])
                #     except:
                #         print(
                #             "Following index out of bound of allowed blocks.", index)
                #     # Update position
                #     position = bounded([x+offset, y+offset, z+offset])
                #     positions.append(position)
                if not index == 0:  # AS INDEX - 1 means air block
                    try:
                        blocks_index.append(index)
                    except:
                        print(
                            "Following index out of bound of allowed blocks.", index)
                    # Update position
                    position = bounded([x+offset[0], y+offset[1], z+offset[2]])
                    positions.append(position)

    #print("Building these block indices:", blocks_index)
    zone = [offset[0], offset[1], offset[2], offset[0]+blocks.shape[0],
            offset[1]+blocks.shape[1], offset[2]+blocks.shape[2]]
    response = client.spawnBlocks(Blocks(blocks=[Block(position=Point(x=int(positions[i][0]), y=int(positions[i][1]), z=int(
            positions[i][2])), type=blocks_index[i], orientation=NORTH) for i in range(len(blocks_index))]))


"""CLEANING MAP POST-SELECTION"""

def clean_positions(positions):
    """
    As a way to clear out a space, place a block of AIR in each of the indicated positions.
    Input:
        positions: np.array of size N*3
    """
    for i in range(positions.shape[0]):
        response = client.spawnBlocks(Blocks(blocks=[Block(position=Point(x=int(positions[i, 0]), y=int(
            positions[i, 1]), z=int(positions[i, 2])), type=AIR, orientation=NORTH)]))
        # print(response)


def clean_batch(positions_batch):
    """
    Cleans a list of positions, by replacing them with air.
    Input:
        positions_batch: list of positions (np.array)

    """
    for positions in positions_batch:
        if len(positions) > 0:
            clean_positions(positions)


def clean_zone(bounds, offset):
    """
    Cleans an area of space within certain bounds, by replacing them by block of AIR.
    Input:
        bounds: dimensions of the zone, list of 3 elements.
        offset: offset position.

    """
    zone = [offset[0], 4, offset[2], offset[0] +
            bounds[0], 4+bounds[1], offset[2]+bounds[2]]
    print("Cleaning the following zone:", zone)
    response = client.fillCube(FillCubeRequest(
        cube=Cube(min=Point(x=int(offset[0]-10), y=int(4), z=int(offset[2]-10)), max=Point(x=int(offset[0]+bounds[0]+10), y=int(
            4+bounds[1]+10), z=int(offset[2]+bounds[2]+10))),  
        type=AIR
    ))
    print(response)

# renders our population by spawning them in on the evocraft server
# how do we render it? for now, just render them in a single row, so it's easy to see which index aligns with which structure
def render_population(model, population):
    generated_structures = generate_from_latent(model, population)
    offset = [0, 0, 0]
    for struc in generated_structures:
        # use evocraft to draw all these into the server.
        build_zone(struc, offset)
        # offset += 10
        offset[0] += 20

# gets users selection as an integer keyboard input
def get_user_input(batch_size, batch_indices, population_size, origin):
    print("Choose one of these structures, from 1 to " +
        str(batch_size) + " (west to east) ")

    # bounds of starting model
    neo_bound = 20
    # Width of entire batch (bound * offset * batch_size)
    width_batch = (30) * batch_size
    perspective = np.floor(width_batch/2)
    rewards = [0]*population_size

    while True: 
        try:
            # index = int(msvcrt.getch)
            index = int(input())
            if index not in np.arange(len(batch_indices))+1:
                print('Entry not valid, try again. Press 0 to EXIT')
            else:
                if index == 0:
                    offset = [
                        origin[0] - np.floor(width_batch / 2), 4, origin[2] - perspective]
                    full_bounds = [(neo_bound + 10) *
                                   batch_size, 20, 20]
                    clean_zone(full_bounds, offset)
                    sys.exit("Bye")
                break
        except:
            print('Entry not valid, try again2')
        # if index == 0:
        #     offset = [
        #         origin[0] - np.floor(width_batch/2), 4, origin[2] - perspective]
        #     full_bounds = [(neo_bound + 10) *
        #                     batch_size, 20, 20]
        #     clean_zone(full_bounds, offset)
        #     sys.exit("Bye")

    return index


# mutates a single vector
def mutate(vector):
    # iterate through each value in the vector
    for i in range(len(vector)):
        # change each value by adding or subtracting mutation_amt with probability mutation_prob
        if random.uniform(0, 1) <= mutation_prob:
            if random.uniform(0, 1) <= .5:
                vector[i] = vector[i] - mutation_amt
            else:
                vector[i] = vector[i] + mutation_amt


# creates a population based on the selected latent vector.
def generate_mutated_population(selected_latent):
    new_pop_list = [selected_latent]
    # generate pop_size - 1 number of new latents vectors, because our selected latent will be added to the next population
    for i in range(pop_size - 1):
        new_latent_vector = np.copy(selected_latent)
        mutate(new_latent_vector)

        new_pop_list.append(new_latent_vector)

    new_pop = np.stack(new_pop_list, axis=0)
    print("shape of new population: ", new_pop.shape)
    return new_pop

# wasserstein loss used for WGAN models. Needed to load WGAN from file
def wasserstein_loss(self, y_true, y_pred):
    return K.mean(y_true * y_pred)



# def train():
def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--generator', type=str, default='GAN',
        metavar='', help='Generator/policy type: VAE, GAN')

    parser.add_argument('--choice_batch', type=int, default=5, metavar='',
        help='Number of structures among which to choose one.')

    parser.add_argument('--position', type=list, default=[0, 10, 0], metavar='',
        help='Initial position for player advised, around which the structures will be evolved.')

    parser.add_argument('--population_size', type=int, default=5, metavar='',
        help='Size of population (needs to be pair and be a multiple of choice_batch or will be approximated).')

    args = parser.parse_args()

    assert args.choice_batch <= args.population_size


    # wipe the slate clean
    bounds = [200, 200, 200]
    offset = [0, 0, 0]
    clean_zone(bounds, offset)

    print("Loading model: ", args.generator)
    ## Two seaparate code blocks depending on which model
    if args.generator == 'GAN':
        # model = load_model(gan_model_path)
        generative_model = load_model(gan_model_path, custom_objects={'wasserstein_loss': wasserstein_loss})

    elif args.generator == 'VAE':
        generative_model = load_model(vae_model_path)

    # print(generative_model.summary())

    # get the size of the latent space of the model
    z_dim = generative_model.layers[0].input_shape[0][1]
    print("latent size: ", z_dim)

    # create a starting population as a list of latent vectors
    population = create_starting_population_gan(args.population_size, z_dim)
    print("starting population shape: ", population.shape)
    # begin interactive evolution loop
    while True:
        # render the population
        render_population(generative_model, population)

        # get user input, which represents the index of the structure they select to further evolve
        # TODO: placeholder. what is this batch index arguement supposed to be?
        batch_indexes = list(range(0,args.population_size))
        selection = get_user_input(args.choice_batch, batch_indexes, population, args.position)

        # get our latent vector to be mutated
        selected_latent = population[selection]

        population = generate_mutated_population(selected_latent)

if __name__ == '__main__':
    main(sys.argv)

