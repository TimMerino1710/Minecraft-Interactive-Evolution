
import sys
sys.path.insert(1, '../3D-VAE/')

from scipy.ndimage import zoom
from visualizer import binary_visualizer
import numpy as np
import json
import os
import glob
import binvox_rw
from pprint import pprint

category_dict = {
"table": "04379243",
"jar": "03593526",
"skateboard": "04225987",
"car": "02958343",
"bottle": "02876657",
"tower": "04460130",
"chair": "03001627",
"bookshelf": "02871439",
"camera": "02942699",
"airplane": "02691156",
"laptop": "03642806",
"basket": "02801938",
"sofa": "04256520",
"knife": "03624134",
"can": "02946921",
"rifle": "04090263",
"train": "04468005",
"pillow": "03938244",
"lamp": "03636649",
"trash bin": "02747177",
"mailbox": "03710193",
"watercraft": "04530566",
"motorbike": "03790512",
"dishwasher": "03207941",
"bench": "02828884",
"pistol": "03948459",
"rocket": "04099429",
"loudspeaker": "03691459",
"file cabinet": "03337140",
"bag": "02773838",
"cabinet": "02933112",
"bed": "02818832",
"birdhouse": "02843684",
"display": "03211117",
"piano": "03928116",
"earphone": "03261776",
"telephone": "04401088",
"stove": "04330267",
"microphone": "03759954",
"bus": "02924116",
"mug": "03797390",
"remote": "04074963",
"bathtub": "02808440",
"bowl": "02880940",
"keyboard": "03085013",
"guitar": "03467517",
"washer": "04554684",
"bicycle": "02834778",
"faucet": "03325088",
"printer": "04004475",
"cap": "02954340",
}

# these are taken from the categories that have over 1000 instances
# TODO: refine categories later
# desired_categories = ['chair', 'bench', 'desk', 'cabinet', 'armchair', 'coffee table,cocktail table', 'straight chair,side chair']
desired_categories = ['table']
SHAPENET_DIR = "H:/ShapeNetCore.v2"

visualizer = binary_visualizer(128, 3, 3)


with open('taxonomy.json') as f:
    taxonomy = json.load(f)



# counts the number of instances of each category
def get_category_counts():
    total = 0
    for item in taxonomy:
        total += item['numInstances']
        if item['numInstances'] > 1000:
            print(item['name'], ": ", item['numInstances'])

    print("total instances: ", total)

# returns a dictionary, where the keys are the names of the categories and the values are a list of filepaths of the binvox models belonging to that cateogry
def get_desired_category_filepaths_dict(name_list):
    directories = get_synset_id_from_name(name_list)
    filepaths = {}
    for dir, category_name in zip(directories, name_list):
        listing = glob.glob(SHAPENET_DIR + "/" + dir + "/*/models/*.solid.binvox")
        filepaths[category_name] = listing
    return filepaths


# synset IDs correspond to a top level directory in the dataset. This will be the list of directories we pull files from
def get_synset_id_from_name(name_list):
    synset_id_list = []
    for name in name_list:
        for item in taxonomy:
            if item['name'] == name:
                synset_id_list.append(item['synsetId'])
    return synset_id_list



def convert_binvox_to_np(binvox_file_dict):
    model_arrays = []
    for category in binvox_file_dict:
        for i, file in enumerate(binvox_file_dict[category]):
            with open(file, 'rb') as f:
                model = binvox_rw.read_as_3d_array(f)
                model_arrays.append(model.data)
                model = zoom(model.data, .5)
                np.save('Shapenet_numpy_files/' + category + "_" + str(i), model)


def get_combined_file(keyword):
    data_list = []
    for file in os.listdir('Shapenet_numpy_files/'):
        if keyword in file:
            f = os.path.join('Shapenet_numpy_files/', file)
            d = np.load(f)
            data_list.append(d)
    data_np = np.asarray(data_list)
    np.save("shapenet_" + keyword + "_combined.npy", data_np)



# get_category_counts()
desired_binxov_files = get_desired_category_filepaths_dict(desired_categories)
convert_binvox_to_np(desired_binxov_files)
get_combined_file("table")
# combined = np.load('shapenet_combined.npy')
# print(combined.shape)
print()