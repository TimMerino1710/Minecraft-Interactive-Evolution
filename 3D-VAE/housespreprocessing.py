import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from nbtschematic import SchematicFile
from pprint import pprint
import json
import tensorflow as tf

# Note:
# nbtschematic does NOT work with .schem files, the newest standard for schematics in worldedit
# .schem files can be converted to the old .schematic file type using https://puregero.github.io/SchemToSchematic/
# however, that site will replace unknown blocks (introduced in later minecraft verisons) with air.
# TODO: fix this issue by reading using nbtlib directly?
def schem_to_np(filepath):
    sf = SchematicFile.load(filepath)
    return sf.blocks.unpack()

def create_combined_data(schematic_dir):
    data_list = []
    for file in os.listdir(schematic_dir):
        if file.endswith('.schematic'):
            file_path = schematic_dir + '/' + file
            file_np = schem_to_np(file_path)
            data_list.append(file_np)

    print(len(data_list))
    combined = np.asarray(data_list)
    print(combined.shape)
    np.save(schematic_dir + "/combined.npy", combined)

create_combined_data('../ingame house schematics/old format schematic files')
