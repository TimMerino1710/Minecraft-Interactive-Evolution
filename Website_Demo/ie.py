# DEEP INTERACTIVE EVOLUTION FOR THE MINECRAFT HOUSE GENERATOR
# written by: Milk


######     LIBRARY IMPORTS     ######

import sys
import yaml
import random
import json
import os
import subprocess

import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.models import load_model
from npy2txt import npy2txt

OG_DIR = str(os.getcwd())


#######     MODEL DEPENDENT FUNCTION DEFINITIONS     #######

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

    return K.categorical_crossentropy(y_true2, y_pred2)

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


#######     TEST FUNCTIONS    #######

# test render and import

compression_list = [[0, 6, 26, 27, 28, 30, 55, 63, 65, 66, 68, 69, 70, 72, 77, 97, 104, 117, 127, 131, 132, 143, 144, 147, 148, 149, 167, 171, 50, 51, 76, 105, 123, 64, 71, 193, 194, 195, 196, 197, 8, 9, 10, 11, 213],[1, 4, 7, 29, 33, 34, 46, 48, 49, 52, 54, 61, 87, 89, 98, 45, 112, 120, 121, 139, 155, 158, 168, 169, 201, 202, 206, 215, 216, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 35, 14, 15, 16, 21, 56, 73, 129, 153],[2, 3, 82, 88, 159, 172, 173, 214, 12, 13, 19, 24, 179, 78, 79, 80, 174],[5, 17, 25, 47, 58, 84, 96, 116, 130, 140, 146, 151, 154, 162],[18, 31, 32, 81, 86, 91, 103, 106, 161, 170, 199, 200, 207],[20, 92, 102, 160, 95],[22, 23, 41, 42, 57, 118, 133, 138, 145, 152, 165],[37, 38, 39, 40, 59, 83, 110, 115, 141, 142, 175],[43, 44, 92, 125, 126, 181, 182, 204, 205],[53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164, 180, 203],[85, 101, 107, 113, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 198]]
real_comp_list = [0,1,2,5,12,18,20,42,53,85,126]   #

def getCompLoc(bi):
	for i in range(len(compression_list)):
		if int(bi) in compression_list[i]:
			return i
	return 0

#remove oob values and compress to the 11 values
def compress_house(house):
	poss_values = sum(compression_list,[])
	alter_house = abs(house)
	for i in range(alter_house.shape[0]):
		for j in range(alter_house.shape[1]):
			for k in range(alter_house.shape[2]):
				# if alter_house[i][j][k] not in poss_values:
				#     alter_house[i][j][k] = 1
				alter_house[i][j][k] = getCompLoc(alter_house[i][j][k])
	alter_house = np.rot90(alter_house,axes=(2,1))
	return alter_house

def importTrainHouses(n=5):
    #get all of the training set
    TRAIN_HOUSES = np.load('/Users/milk/Desktop/GIL_Lab/MC_int_evo/Minecraft-Interactive-Evolution/ingame_house_schematics/old_format_schematic_files/combined.npy',allow_pickle=True)
    fin_houses = []
    # for h in random.choices(TRAIN_HOUSES,k=n):
    hs = [37, 51, 75, 38, 65]
    for hi in hs:
        h = TRAIN_HOUSES[hi]
        h = np.array(h).astype(int)

        # houses look rotated... just rotate them back
        h = np.rot90(h,axes=(0,2))
        
        # remove bottom layer (got the ground as well) - i can't believe i got it right on the first try...
        h = h[3:, 3:, 1:-2]

        #rotate again?                   8                  
        h2 = np.rot90(h,axes=(-1,1))
        h2 = np.rot90(h2,axes=(-1,1))

        # compress the house
        h3 = compress_house(h2)
        fin_houses.append(centerHouse(h3))
        # fin_houses.append(h3)

    return fin_houses
    


########    MODEL IMPORTS     ########   

# read in the yaml file and set the parameters
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# load the models
generator = load_model(CONFIG['generator'],custom_objects={'Sampling': Sampling})

C_WEIGHTS = K.variable(np.array([1e-5,1,1,1,1,1,1,1,1,1,1]))
sml = super_mask_loss(C_WEIGHTS)
painter = load_model(CONFIG['painter'],custom_objects={'loss': sml})


########   HELPER FUNCTIONS  ########

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

#paint the houses and apply the masking
def paintHouses(houses):
    paint = painter.predict(houses).squeeze()
    # print(np.unique(unencodeHouse(paint)))
    fin_houses = []
    for pi in range(len(houses)):
        mask_pred = np.array(unencodeHouse(paint[pi])*houses[pi])
        fin_houses.append(mask_pred)
    return np.array(fin_houses)


#centers a house in the middle of the space
def centerHouse(house,dim=[16,16,16]):
    #find the bounds of the house
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





########    INT EVO CODE     ########


# create randomized latent vectors to start the population
def initSamples(n=5):
    Z = []
    for i in range(n):
        z = np.random.normal(0,1,(1,CONFIG['latent_dim']))
        Z.append(z)
    return Z

def randTrainSamples(n=5):
    Z = []
    with open("models/encoded_train_houses.txt") as f:
        lines = f.readlines()
        Z = random.choices(lines,k=n)
        Z = [np.expand_dims(np.array(zi.split(",")).astype(float),axis=0) for zi in Z]
    return np.array(Z)

# generate, paint, and render the samples using the latent vectors
def genHouses(Z):
    # generate the houses
    out_houses = []
    for z in Z:
        # generate the house (and reduce to half size)
        bin_house = np.around(generator.predict(z).squeeze())

        #make the house smaller (16,16,16)
        small_house = bin_house.copy()
        while(small_house.shape[0]>16):
            small_house = halfHouse(small_house)
        out_houses.append(small_house)

        # print(half_house)

    
    # paint the houses and return final
    out_houses = [np.rot90(h,2,axes=(0,1)) for h in out_houses]
    out_houses = np.array(out_houses)
    paint_houses = paintHouses(out_houses)
    paint_houses = paint_houses.astype(int)  #cast as integer array
    trans_houses = [np.rot90(h,2,axes=(0,1)) for h in paint_houses]

    # ROTATE + center (do not change)
    # trans_houses = [h for h in paint_houses]
    trans_houses = [np.rot90(h,3,axes=(-1,1)) for h in trans_houses]
    trans_houses = [centerHouse(h) for h in trans_houses]
    
    return trans_houses


# make a new population from the best selected z latent vector
# same algorithm used in "Deep Interactive Evolution" paper: https://arxiv.org/abs/1801.08230
def mutatePop(bestZ,n=5):
    new_pop = []
    bzs = bestZ.squeeze()

    #add the original back
    new_pop.append(bzs)

    #mutate the latent vector
    for i in range(n-2):
        newZ = bzs + np.random.normal(0, 1, bzs.shape)
        new_pop.append(newZ)

    #add a random latent vector + reshape
    new_pop.append(np.random.normal(0, 1, bzs.shape))
    new_pop = np.array(new_pop)
    np.random.shuffle(new_pop)
    new_pop = np.expand_dims(new_pop,axis=1)

    return np.array(new_pop)
    

# make a new population from the best selected z latent vectors using crossover and mutation
# same algorithm used in "Deep Interactive Evolution" paper: https://arxiv.org/abs/1801.08230
def multiMutatePop(bestZs,n=5):
    new_pop = []

    #add one of the originals back
    new_pop.append(random.choice(bestZs).squeeze())

    #crossover the latent vectors
    for i in range(n-2):
        #pick 2 parents
        if len(bestZs) == 2:
            parents = bestZs.squeeze()
        else:
            parents = bestZs[np.random.choice(list(range(len(bestZs))), 2, replace=False)].squeeze()
        
        #select a random point from each parent
        newZ = []
        for i in range(len(parents[0])):
            zi = random.choice(parents)[i]
            newZ.append(zi)
        new_pop.append(newZ)
    
    #mutate the latent vectors
    for z in new_pop[1:]:
        z = np.array(z) + np.random.normal(0, 1, len(z))
    
    #add a random latent vector
    new_pop.append(np.random.normal(0, 1, len(new_pop[0])))

    #shuffle the population + add a dimension back
    new_pop = np.array(new_pop)
    np.random.shuffle(new_pop)
    new_pop = np.expand_dims(new_pop,axis=1)

    return new_pop
    

#############  MAIN  #############
     
#inital evolution step
def startEvo():
    zpop = initSamples(CONFIG['num_samples'])
    # zpop = randTrainSamples(CONFIG['num_samples'])
    houses = genHouses(zpop)
    # houses = importTrainHouses(CONFIG['num_samples'])
    # houses = [[[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,3,8,0,0,0,0,0,0],[0,0,0,0,0,0,0,8,8,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,3,0,0,1,1,3,0,0,0,0,0],[0,0,0,0,0,3,3,0,1,3,0,0,0,0,0,0],[0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0],[0,0,0,0,0,0,3,3,9,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,3,2,2,2,1,1,1,1,0,0,0,0],[0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,2,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,1,2,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,8,1,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],[0,0,0,0,3,3,1,1,0,3,3,0,0,0,0,0],[0,0,0,0,3,3,3,0,0,3,0,0,0,0,0,0],[0,0,0,0,0,3,3,0,3,0,0,0,0,0,0,0],[0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,3,2,2,2,2,2,1,0,0,0,0,0],[0,0,0,0,3,2,2,1,3,1,0,0,0,0,0,0],[0,0,0,0,0,0,8,8,3,2,0,0,0,0,0,0],[0,0,0,0,0,0,1,8,2,2,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,8,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,3,3,3,0,0,0,3,3,0,0,0,0,0],[0,0,0,0,3,3,0,0,0,3,0,0,0,0,0,0],[0,0,0,0,0,3,9,0,3,0,0,0,0,0,0,0],[0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,3,2,2,2,2,1,3,0,0,0,0,0],[0,0,0,0,2,2,2,0,0,3,0,0,0,0,0,0],[0,0,0,0,0,1,8,0,0,2,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,8,2,0,0,0,0,0,0],[0,0,0,0,0,1,1,10,8,2,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],[0,0,0,3,3,3,0,0,0,3,3,0,0,0,0,0],[0,0,0,0,3,9,0,0,0,9,0,0,0,0,0,0],[0,0,0,0,0,3,1,0,3,0,0,0,0,0,0,0],[0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,3,2,2,2,2,1,0,0,0,0,0,0],[0,0,0,0,2,3,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,8,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],[0,0,0,3,3,1,0,0,0,1,9,0,0,0,0,0],[0,0,0,0,3,3,0,0,0,9,0,0,0,0,0,0],[0,0,0,0,0,3,0,0,3,0,0,0,0,0,0,0],[0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,8,2,2,2,2,2,2,0,0,0,0,0],[0,0,0,0,4,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,8,8,0,3,0,0,0,0,0,0],[0,0,0,0,0,1,0,1,0,3,0,0,0,0,0,0],[0,0,0,0,0,3,0,0,0,5,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,3,3,0,0,0,0,3,9,0,0,0,0,0],[0,0,0,0,3,0,0,0,0,9,0,0,0,0,0,0],[0,0,0,0,0,3,0,0,3,0,0,0,0,0,0,0],[0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,8,2,2,2,2,2,2,0,0,0,0,0],[0,0,0,0,4,2,0,0,8,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,2,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,1,0,3,0,3,0,0,0,0,0,0],[0,0,0,0,1,3,0,0,0,3,0,0,0,0,0,0],[0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,3,3,0,0,0,0,3,9,0,0,0,0,0],[0,0,0,0,3,0,0,0,3,9,0,0,0,0,0,0],[0,0,0,0,0,3,0,0,3,0,0,0,0,0,0,0],[0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,2,2,2,2,2,2,2,0,0,0,0,0],[0,0,0,0,4,1,2,2,2,0,0,0,0,0,0,0],[0,0,0,0,0,3,0,1,2,0,0,0,0,0,0,0],[0,0,0,0,0,2,2,2,1,0,0,0,0,0,0,0],[0,0,0,0,3,1,1,1,1,0,0,0,0,0,0,0],[0,0,0,0,3,1,1,1,1,0,0,0,0,0,0,0],[0,0,0,0,3,1,1,1,3,0,0,0,0,0,0,0],[0,0,0,3,3,3,1,1,1,3,9,0,0,0,0,0],[0,0,0,0,3,3,3,3,3,2,0,0,0,0,0,0],[0,0,0,0,0,3,3,3,2,0,0,0,0,0,0,0],[0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,2,2,2,2,2,0,8,2,0,0,0,0,0],[0,0,0,3,3,0,2,8,0,0,4,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,3,9,0,0,0,0,3,8,0,0,0,0,0],[0,0,0,0,3,9,0,0,3,8,0,0,0,0,0,0],[0,0,0,0,0,3,9,2,2,0,0,0,0,0,0,0],[0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,3,3,2,2,0,8,8,0,0,0,0,0],[0,0,0,0,0,0,3,3,0,8,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]]*5

    return zpop, houses

#next step of the evolution process for a single z vector
def nextEvo(best_z):
    zpop = mutatePop(best_z, CONFIG['num_samples'])
    houses = genHouses(zpop)

    return zpop, houses

#next step of the evolution process for multiple z vectors (uses crossover)
def multiNextEvo(best_zs):
    zpop = multiMutatePop(best_zs, CONFIG['num_samples'])
    houses = genHouses(zpop)

    return zpop, houses



if __name__ == '__main__':
    #inital evolution step
    print("initial")
    zpop, houses = startEvo()

    #evolve one vector
    print("evolve 1 vector")
    print(zpop[0].shape)
    zpop, houses = nextEvo(zpop[0])

    #evolve random parents
    print("evolve 3 vectors")
    zpop = np.array(zpop)
    si = np.random.choice(list(range(len(zpop))),3,replace=False)
    selected = zpop[si]
    print(selected.shape)
    zpop2, houses2 = multiNextEvo(selected)

