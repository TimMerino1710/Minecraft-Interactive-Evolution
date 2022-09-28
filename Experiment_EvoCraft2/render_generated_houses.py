from bz2 import compress
import tensorflow as tf
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

channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

# compression_list = [[0, 6, 26, 27, 28, 30, 55, 63, 65, 66, 68, 69, 70, 72, 77, 97, 104, 117, 127, 131, 132, 143, 144, 147, 148, 149, 167, 171, 50, 51, 76, 105, 123],[1, 4, 7, 29, 33, 34, 46, 48, 49, 52, 54, 61, 87, 89, 98, 45, 112, 120, 121, 139, 155, 158, 168, 169, 201, 202, 206, 215, 216, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 35, 14, 15, 16, 21, 56, 73, 129, 153],[2, 3, 82, 88, 159, 172, 173, 214],[5, 17, 25, 47, 58, 84, 96, 116, 130, 140, 146, 151, 154, 162],[8, 9, 10, 11, 213],[12, 13, 19, 24, 179],[18, 31, 32, 81, 86, 91, 103, 106, 161, 170, 199, 200, 207],[20, 92, 102, 160, 95],[22, 23, 41, 42, 57, 118, 133, 138, 145, 152, 165],[37, 38, 39, 40, 59, 83, 110, 115, 141, 142, 175],[43, 44, 92, 125, 126, 181, 182, 204, 205],[53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164, 180, 203],[64, 71, 193, 194, 195, 196, 197],[78, 79, 80, 174],[85, 101, 107, 113, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 198],]
compression_list = [[0, 6, 26, 27, 28, 30, 55, 63, 65, 66, 68, 69, 70, 72, 77, 97, 104, 117, 127, 131, 132, 143, 144, 147, 148, 149, 167, 171, 50, 51, 76, 105, 123, 64, 71, 193, 194, 195, 196, 197, 8, 9, 10, 11, 213],[1, 4, 7, 29, 33, 34, 46, 48, 49, 52, 54, 61, 87, 89, 98, 45, 112, 120, 121, 139, 155, 158, 168, 169, 201, 202, 206, 215, 216, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 35, 14, 15, 16, 21, 56, 73, 129, 153],[2, 3, 82, 88, 159, 172, 173, 214, 12, 13, 19, 24, 179, 78, 79, 80, 174],[5, 17, 25, 47, 58, 84, 96, 116, 130, 140, 146, 151, 154, 162],[18, 31, 32, 81, 86, 91, 103, 106, 161, 170, 199, 200, 207],[20, 92, 102, 160, 95],[22, 23, 41, 42, 57, 118, 133, 138, 145, 152, 165],[37, 38, 39, 40, 59, 83, 110, 115, 141, 142, 175],[43, 44, 92, 125, 126, 181, 182, 204, 205],[53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164, 180, 203],[85, 101, 107, 113, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 198]]
id_to_type = {0:'AIR', 1:'STONE', 2:'GRASS', 5:'PLANKS', 8:'WATER', 12:'SAND', 18:'LEAVES', 20:'GLASS', 22:'LAPIS_BLOCK', 37:'YELLOW_FLOWER', 43:'STONE_SLAB', 53:'OAK_STAIRS', 85:'FENCE'}


BOUNDS_WORLD = [[-30000000, 29999999],  [4, 255], [-30000000, 29999999]]

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
                if index not in blockid_to_type:
                    for compressed in compression_list:
                        if index in compressed:
                            index = compressed[0]
                            break
                """
                for compressed in compression_list:
                    if index in compressed:
                        index = compressed[0]
                        break
                """
                blocks_index.append(index)
                position = bounded([x+offset[0], y+offset[1], z+offset[2]])
                positions.append(position)

    zone = [offset[0], offset[1], offset[2], offset[0]+blocks.shape[0],
            offset[1]+blocks.shape[1], offset[2]+blocks.shape[2]]
    response = client.spawnBlocks(Blocks(blocks=[Block(position=Point(x=int(positions[i][0]), y=int(positions[i][1]), z=int(
            positions[i][2])), type=blockid_to_type[blocks_index[i]], orientation=NORTH) for i in range(len(blocks_index))]))

    return blocks_index


# renders our population by spawning them in on the evocraft server
def render_house_set(houses,offset=None):
    if offset == None:
        offset = [0, 4, 0]
    for struc in houses:
        # use evocraft to draw all these into the server.
        rendered_struc = build_zone(struc, offset)
        offset[0] += struc.shape[0]+4






class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        mean_mu, log_var = inputs
        epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0., stddev=1.) 
        return mean_mu + K.exp(log_var/2)*epsilon

def mask_loss(y_true,y_pred):
    #cast values
    zero = tf.constant(0, dtype=tf.float32)
    y_true2 = tf.cast(y_true,tf.float32)
    y_pred2 = tf.cast(y_pred,tf.float32)
    
    #apply mask
    mask = tf.cast(tf.where(tf.not_equal(y_true2, zero),1,0),tf.float32)
    mask_pred = tf.math.multiply(y_pred2,mask)

    return tf.losses.mean_squared_error(y_true2,mask_pred)


# generate and show new sample from the VAE
def generateVAESamples(vae_model,n=8,zsize=100):
    vae_samples = np.around(vae_model.predict(np.random.normal(0,1,size=(n,zsize))).squeeze())
    return vae_samples

def paintHouses(houses,painter_model):
    return painter_model.predict(houses).squeeze()


#find most common element in a 3d array
def mostComm3d(a):
  ct = {}
  for r in a:
    for c in r:
      for d in c:
        di = int(d)
        if di not in ct:
          ct[di] = 0
        ct[di] += 1
  return max(ct, key=ct.get)


#reduce the size of a house by half (take majority in 2x2 area)
def halfHouse(h):
  hh = np.zeros(shape=(int(h.shape[0]/2),int(h.shape[1]/2),int(h.shape[2]/2)))
  for x in range(hh.shape[0]):
    for y in range(hh.shape[1]):
      for z in range(hh.shape[2]):
        #set as majority in the area
        ss = h[x*2:(x+1)*2,y*2:(y+1)*2,z*2:(z+1)*2]
        mblock = mostComm3d(ss)
        hh[x][y][z] = mblock
  return hh


#run the whole thing    
if __name__ == "__main__":
    # wipe the slate clean
    bounds = [200, 200, 200]
    offset = [0, 0, 0]
    clean_zone(bounds, offset)

    #import the models
    vae = load_model(f"../beta_models/decoder-20_z100.h5",custom_objects={'Sampling': Sampling})
    painter = load_model(f"../beta_models/painter-20ep.h5",custom_objects={'mask_loss': mask_loss})

    vae.summary()
    painter.summary()

    #generate some samples and paint them
    bin_samples = np.expand_dims(generateVAESamples(vae,5),axis=-1)
    half_samples = np.array([halfHouse(bh) for bh in bin_samples])
    paint_samples = paintHouses(half_samples,painter)
    print(bin_samples.shape)
    print(paint_samples.shape)

    #render the binary and painted houses on the other side
    print(f"-- Rendering generated houses -- ")
    rot_samp = np.array([np.rot90(hq,axes=(2,1)) for hq in bin_samples])
    render_house_set(rot_samp)
    #render_house_set(rot_samp,[0,4,0])







