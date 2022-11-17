import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
from tqdm import tqdm


from json import JSONEncoder
import json

class NumpyArrayEncoder(JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return JSONEncoder.default(self, obj)

# directly convert a npy array to a JSON string
def npy2txt(arr):
	return json.dumps(arr, cls=NumpyArrayEncoder)




class Sampling(tf.keras.layers.Layer):
	"""Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
	def call(self, inputs):
		mean_mu, log_var = inputs
		epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0., stddev=1.) 
		return mean_mu + K.exp(log_var/2)*epsilon


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

#scale up the house (double each block)
def doubleHouse(h):
  dh = np.zeros(shape=(h.shape[0]*2,h.shape[1]*2,h.shape[2]*2))
  m = [(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(0,1,1),(1,0,1),(1,1,1)]
  for x in range(h.shape[0]):
	  for y in range(h.shape[1]):
		  for z in range(h.shape[2]):
			  v = h[x][y][z]
			  if v == 0:
				  continue
			  for mi in m:
				  dh[x*2+mi[0]][y*2+mi[1]][z*2+mi[2]] = v
  return dh

#turn a house to binary
def binHouse(h):
	idx = np.nonzero(h)
	hb = np.zeros(shape=h.shape)
	for i in range(len(idx[0])):
		a,b,c = idx
		hb[a[i]][b[i]][c[i]] = 1
	return hb


#import the models
VAE_MODEL_LOC = "../beta_models/z-100_d-1024"
VAE_MODEL_VER = "-50_beta"

vae_enc = load_model(f"{VAE_MODEL_LOC}/encoder{VAE_MODEL_VER}.h5",custom_objects={'Sampling': Sampling})
# vae_dec = load_model(f"{VAE_MODEL_LOC}/decoder{VAE_MODEL_VER}.h5",custom_objects={'Sampling': Sampling})

#get all of the training set
TRAIN_HOUSES = np.load('../ingame_house_schematics/old_format_schematic_files/combined.npy',allow_pickle=True)
houses = []
house_txt = []
with tqdm(total=len(TRAIN_HOUSES)) as pbar:
	for h in TRAIN_HOUSES:
		h = np.array(h).astype(int)

		# houses look rotated... just rotate them back
		h = np.rot90(h,axes=(0,2))
		
		# remove bottom layer (got the ground as well) - i can't believe i got it right on the first try...
		h = h[3:, 3:, 1:-2]

		#rotate again?                   8                  
		# h2 = np.rot90(h,axes=(-1,1))
		# h2 = np.rot90(h2,axes=(-1,1))
		# h2 = np.rot90(h2,axes=(-1,1))

		#binary house
		bh = binHouse(h)

		house_txt.append(bh)

		#double it
		h2 = doubleHouse(bh)

		#change oob values
		# alter_house = compress_house(h2)
		
		houses.append(h2)

		pbar.update(1)

print(np.array(houses).shape)

#export the binary houses
with open("/Users/milk/Desktop/bin_house.txt", "w+") as f:
	for h in house_txt:
		f.write(f"{npy2txt(np.array(h).astype(int))}\n\n")


# encode the whole output
enc_train = []
for h in houses:
	z = np.array(vae_enc.predict(np.array([h]),verbose=False)[2]).squeeze()
	enc_train.append(z)

#export to a file
with open("/Users/milk/Desktop/MCIntEvo-Demo/models/encoded_train_houses.txt", "w+") as f:
	for z in enc_train:
		zstr = [str(zi) for zi in z]
		#print(zstr)
		f.write(f'{",".join(zstr)}\n')




