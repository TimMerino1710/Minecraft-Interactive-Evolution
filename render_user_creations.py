
import numpy as np
import os
import random
import sys
import argparse
from nbtschematic import SchematicFile
from random import randint

user_goal_houses = np.load('user_goal_houses.npy')
user_open_houses = np.load('user_open_houses.npy')

compression_list = [0, 1, 3, 5, 44]
outdir = './UserSchematics/'
os.makedirs(outdir, exist_ok = True)

def replace_with_blockid(house):
    for i in range(0, 5):
        house[house == i] = compression_list[i]

def write_house_to_schem(house, schem_loc):
    sf = SchematicFile(shape=(16,16,16))
    for index, block_id in np.ndenumerate(house):
        sf.blocks[index[1], index[2], index[0]] = block_id
    sf.save(schem_loc)

for i, goal_house in enumerate(user_goal_houses):
    replace_with_blockid(goal_house)
    write_house_to_schem(goal_house, outdir + "goal_house_" + str(i) + ".schematic")


for i, open_house in enumerate(user_open_houses):
    replace_with_blockid(open_house)
    write_house_to_schem(open_house, outdir + "open_house_" + str(i) + ".schematic")
