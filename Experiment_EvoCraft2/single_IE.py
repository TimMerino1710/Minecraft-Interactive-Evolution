# REGENERATES HOUSES FROM VAE TO SHOW SIDE BY SIDE

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
import yaml
import time

#import the configuation file
with open('ie_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# custom functions - for the VAE
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        mean_mu, log_var = inputs
        epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0., stddev=1.) 
        return mean_mu + K.exp(log_var/2)*epsilon

#custom functions - for the Painter
def super_mask_loss(weights):
    weights = K.variable(weights)
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss






channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

# compression_list = [[0, 6, 26, 27, 28, 30, 55, 63, 65, 66, 68, 69, 70, 72, 77, 97, 104, 117, 127, 131, 132, 143, 144, 147, 148, 149, 167, 171, 50, 51, 76, 105, 123],[1, 4, 7, 29, 33, 34, 46, 48, 49, 52, 54, 61, 87, 89, 98, 45, 112, 120, 121, 139, 155, 158, 168, 169, 201, 202, 206, 215, 216, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 35, 14, 15, 16, 21, 56, 73, 129, 153],[2, 3, 82, 88, 159, 172, 173, 214],[5, 17, 25, 47, 58, 84, 96, 116, 130, 140, 146, 151, 154, 162],[8, 9, 10, 11, 213],[12, 13, 19, 24, 179],[18, 31, 32, 81, 86, 91, 103, 106, 161, 170, 199, 200, 207],[20, 92, 102, 160, 95],[22, 23, 41, 42, 57, 118, 133, 138, 145, 152, 165],[37, 38, 39, 40, 59, 83, 110, 115, 141, 142, 175],[43, 44, 92, 125, 126, 181, 182, 204, 205],[53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164, 180, 203],[64, 71, 193, 194, 195, 196, 197],[78, 79, 80, 174],[85, 101, 107, 113, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 198],]
compression_list = [[0, 6, 26, 27, 28, 30, 55, 63, 65, 66, 68, 69, 70, 72, 77, 97, 104, 117, 127, 131, 132, 143, 144, 147, 148, 149, 167, 171, 50, 51, 76, 105, 123, 64, 71, 193, 194, 195, 196, 197, 8, 9, 10, 11, 213],[1, 4, 7, 29, 33, 34, 46, 48, 49, 52, 54, 61, 87, 89, 98, 45, 112, 120, 121, 139, 155, 158, 168, 169, 201, 202, 206, 215, 216, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 35, 14, 15, 16, 21, 56, 73, 129, 153],[2, 3, 82, 88, 159, 172, 173, 214, 12, 13, 19, 24, 179, 78, 79, 80, 174],[5, 17, 25, 47, 58, 84, 96, 116, 130, 140, 146, 151, 154, 162],[18, 31, 32, 81, 86, 91, 103, 106, 161, 170, 199, 200, 207],[20, 92, 102, 160, 95],[22, 23, 41, 42, 57, 118, 133, 138, 145, 152, 165],[37, 38, 39, 40, 59, 83, 110, 115, 141, 142, 175],[43, 44, 92, 125, 126, 181, 182, 204, 205],[53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164, 180, 203],[85, 101, 107, 113, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 198]]
real_comp_list = [0,1,2,5,12,18,20,42,53,85,126]   #air,stone,dirt,wood,sand,leaves,glass,metal,stairs,fence,slab

#import the training data houses
# TRAIN_HOUSES = np.load('../ingame_house_schematics/old_format_schematic_files/combined.npy',allow_pickle=True)

BOUNDS_WORLD = [[-30000000, 29999999],  [4, 255], [-30000000, 29999999]]



##########  EVOCRAFT FUNCTIONS  ##########


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

#render a house @ the center position in front of the player
def renderHouse(struc,offset=None):
    '''
        Render a house at the center position in front of the player
        Input:
            struc: a house (painted)
            offset: offset position
    '''
    #move the house by offset
    if offset == None:
        offset = [0, 4, 0]
    # use evocraft to draw all these into the server.
    rendered_struc = build_zone(struc, offset)

def resetHouse():
    # wipe the slate clean
    bounds = [100, 100, 100]
    offset = [-50, 4, 50]
    clean_zone(bounds, offset)





#######     MODEL FUNCTIONS     #######


# generate binary samples from the generator using a normal latent vector
def normalHouseGen(gen_model,n=1,zsize=100):
    gen_samples = np.around(gen_model.predict(np.random.normal(0,1,size=(n,zsize)),verbose=False))
    # return np.expand_dims(gen_samples,axis=-1)
    return gen_samples

#paint the houses and apply the masking
def paintHouse(house,painter_model):
    paint = painter_model.predict(house,verbose=False)
    print(">>>> ", paint.shape)
    mask_pred = np.array(unencodeHouse(paint)*house)
    return np.array(mask_pred)

#generate and paint a house from an latent vector input
def houseLVE(gen_model,painter_model,lve):
    #make a new house and paint it
    gen_h = np.around(gen_model.predict(lve,verbose=False))
    half_h = halfHouse(gen_h[0])
    paint_h = paintHouse(np.expand_dims(half_h,axis=0),painter_model).squeeze()
    
    #recenter
    crop_h = cropHouse(paint_h)
    print("CROP: ", crop_h.shape)

    center_h = centerHouse(crop_h)
    
    return center_h

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

#un-one hot encode the house
def unencodeHouse(h):
    return np.argmax(h,axis=-1)

def cropHouse(h):
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(h)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)
    out = h[top_left[0]:bottom_right[0]+1,  # plus 1 because slice isn't
              top_left[1]:bottom_right[1]+1,top_left[2]:bottom_right[2]+1]  # inclusive
    return out

# place house in the center of a space (written by Copilot)
def centerHouse(h,bounds=[16,16,16]):
    #find the center of the space
    # center = [int(bounds[0]/2),int(bounds[1]/2),int(bounds[2]/2)]
    center = [int(bounds[0]/2),int(bounds[1]/2),int(bounds[2])]
    
    #find the center of the house
    hcenter = [int(h.shape[0]/2),int(h.shape[1]/2),int(h.shape[2]/2)]
    
    #find the offset
    offset = [center[0]-hcenter[0],center[1]-hcenter[1],center[2]-hcenter[2]]
    
    print(offset)

    #place the house in the center
    ch = np.zeros(shape=bounds)
    ch[offset[0]:offset[0]+h.shape[0],offset[1]:offset[1]+h.shape[1],0:h.shape[2]] = h
    
    return ch


def reassignBlock(house):
    alter_house = np.copy(house)
    for i in range(alter_house.shape[0]):
        for j in range(alter_house.shape[1]):
            for k in range(alter_house.shape[2]):
                alter_house[i][j][k] = real_comp_list[int(alter_house[i][j][k])]
    alter_house = np.rot90(alter_house,axes=(2,1))
    return alter_house




#######     MAIN FUNCTIONS     #######


def main():
    '''
        Main function to run the interactive evolution
    '''
    resetHouse()

    #import the generator
    if config['gen_type'] == "VAE":
        generator = load_model(config['gen_loc'],custom_objects={'Sampling': Sampling})
    
    #import the painter
    if config['paint_loss'] == "super":
        C_WEIGHTS = K.variable(np.array([1e-5,1,1,1,1,1,1,1,1,1,1]))
        sml = super_mask_loss(C_WEIGHTS)
        painter = load_model(config['paint_loc'],custom_objects={'loss': sml})

    #generate a single sample and shrink
    # binary_h = normalHouseGen(generator,n=1,zsize=config['gen_z'])
    # print("Generated house: ",binary_h.shape) #should be 32x32x32x1
    # half_h = halfHouse(binary_h[0])   
    # print("Half house: ",half_h.shape)  #should be 16x16x16
    
    #paint the samples
    # paint_h = paintHouse(np.expand_dims(half_h,axis=0),painter)
    # print(paint_h.shape)  #should be 16x16x16

    #make a new house from a starter latent vector


    Z = np.random.normal(0,1,size=(1,config['gen_z']))

    #evolve a latent vector for a house
    for i in range(5):
        #clear the area
        resetHouse()

        #make a new house
        house = houseLVE(generator,painter,Z)

        #render a house
        renderHouse(reassignBlock(house),offset=[-7,4,-22])

        time.sleep(1)

        #change the latent vector
        Z = np.random.normal(0,1,size=(1,config['gen_z']))
    

if __name__ == "__main__":
    main()