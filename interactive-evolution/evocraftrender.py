from bz2 import compress
from keras.models import load_model
import keras.backend as K
import numpy as np
import pandas as pd
import random
import grpc
import minecraft_pb2_grpc
from minecraft_pb2 import *
import sys
import argparse
from blockid_to_type import blockid_to_type

# connect to minecraft
channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

# mapping of blocks to compressed block categories
mapping_df = pd.read_csv('../compression_csv.csv')
id_to_type = {0:'AIR', 1:'STONE', 2:'GRASS', 5:'PLANKS', 8:'WATER', 12:'SAND', 18:'LEAVES', 20:'GLASS', 22:'LAPIS_BLOCK', 37:'YELLOW_FLOWER', 43:'STONE_SLAB', 53:'OAK_STAIRS', 85:'FENCE'}

# This is hacky, but because we sort when we map the compressed values, we can decompress them by using the models output as an index for this list. 0 will always be air, 8 will always be glass (or w/e), etc
# compressed_blocks = [0, 1, 2, 5, 12, 18, 20, 53, 85, 126]


BOUNDS_WORLD = [[-30000000, 29999999],  [4, 255], [-30000000, 29999999]]
DIRECTIONS = ["N", "W", "S", "E", "U", "D"]
DIRECTIONS_2D = ["W", "E", "U", "D"]
ORIENTATIONS = ["O.N", "O.W", "O.S", "O.E", "O.U", "O.D"]


class EvoRenderer():
    def __init__(self, model, model_type, bin_or_cat, struct_num, offset_x=20, offset_y=20):
        self.model = model
        self.struct_num = struct_num
        self.model_type = model_type
        self.bin_or_cat = bin_or_cat
        self.latent_size = model.layers[0].input_shape[0][1]
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.compressed_blocks = [0, 1, 2, 5, 12, 18, 20, 53, 85, 126]
        self.offset = [0,0,0]
        #TODO: get the size of the structure we're using by looking at the last layer of our generative model


        # Evocraft variables
        self.BOUNDS_WORLD = [[-30000000, 29999999], [4, 255], [-30000000, 29999999]]
        self.DIRECTIONS = ["N", "W", "S", "E", "U", "D"]
        self.DIRECTIONS_2D = ["W", "E", "U", "D"]
        self.ORIENTATIONS = ["O.N", "O.W", "O.S", "O.E", "O.U", "O.D"]

        # clear map
        self.clean_zone([200, 200, 200], [0, 0, 0])

        # read the mapping dataframe
        self.mapping_df = pd.read_csv('../compression_csv.csv')

        # connect to minecraft server
        self.channel = grpc.insecure_channel('localhost:5001')
        self.client = minecraft_pb2_grpc.MinecraftServiceStub(channel)



    def decompress(self, gen_structs):
        # turns integer values back into block IDs by inverting our compression
        for val in np.unique(gen_structs):
            gen_structs[gen_structs == val] = self.compressed_blocks[val]

        return gen_structs


    # take latent vectors, and generate structures from them
    def generate_from_latent_categorical(self, model, latent_vectors):
        generated_structures = model.predict(latent_vectors)

        # a categorical GAN will output one-hot encoded, so this must turned back into categorical
        generated_structures = np.argmax(generated_structures, axis=4)

        return generated_structures

        # take latent vectors, and generate structures from them

    def generate_from_latent_binary(self, model, latent_vectors):
        generated_structures = model.predict(latent_vectors)
        return generated_structures


    # generates an array of random normal latent vectors
    def generate_latents_rn(self):
        return np.random.normal(0, 1, (self.struct_num, self.latent_size))


    """RENDERING HELPERS"""
    def bound_coordinate(self, value, coord):
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

    def bounded(self, position):
        """
        Bounds the position according to BOUNDS_WORLD.
        INPUT:
            position: a 3D position.

        OUTPUT:
            bounded_position: a 3D position, within the boundaries given by BOUNDS_WORLD

        """
        bounded_position = [self.bound_coordinate(position[0], 0), self.bound_coordinate(
            position[1], 1), self.bound_coordinate(position[2], 2)]
        return bounded_position

    def build_compression_list(self, mapping_df):
        pass

    # spawns a building
    def build_zone(self, blocks, offset):
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

                    # get block type (will be a category from 0 to 9)
                    block_id = int(blocks[x, y, z])

                    # hacky decompression
                    block_id = self.compressed_blocks[block_id]

                    blocks_index.append(block_id)
                    position = self.bounded([x+offset[0], y+offset[1], z+offset[2]])
                    positions.append(position)

        zone = [offset[0], offset[1], offset[2], offset[0]+blocks.shape[0],
                offset[1]+blocks.shape[1], offset[2]+blocks.shape[2]]
        response = client.spawnBlocks(Blocks(blocks=[Block(position=Point(x=int(positions[i][0]), y=int(positions[i][1]), z=int(
                positions[i][2])), type=blockid_to_type[blocks_index[i]], orientation=NORTH) for i in range(len(blocks_index))]))

        return blocks_index


    #TODO: everything blow this
    """CLEANING MAP POST-SELECTION"""
    def clean_positions(self, positions):
        """
        As a way to clear out a space, place a block of AIR in each of the indicated positions.
        Input:
            positions: np.array of size N*3
        """
        for i in range(positions.shape[0]):
            response = client.spawnBlocks(Blocks(blocks=[Block(position=Point(x=int(positions[i, 0]), y=int(
                positions[i, 1]), z=int(positions[i, 2])), type=AIR, orientation=NORTH)]))


    def clean_zone(self, bounds, offset):
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

    # wasserstein loss used for WGAN models. Needed to load WGAN from file
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def render(self, model):
        print("rendering structures...")
        latents = self.generate_latents_rn()
        if self.bin_or_cat == "bin":
            structs = self.generate_from_latent_categorical(model, latents)
        else:
            structs = self.generate_from_latent_categorical(model, latents)

        for struct in structs:
            # use evocraft to draw all these into the server.
            rendered_struc = self.build_zone(struct, self.offset)
            self.offset[0] += 20
        self.offset[0] = 0
        self.offset[2] += -25


