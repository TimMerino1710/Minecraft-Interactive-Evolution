import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from nbtschematic import SchematicFile
import importlib

# Takes a directory of .schem files, and converts them into a combined array of size (# samples, x, y, z), where each entry is a minecraft block name (ex: minecraft:air)
def create_combined_blockname_data(schematic_dir):
    data_list = []
    for file in os.listdir(schematic_dir):
        if file.endswith('.schem'):
            file_path = schematic_dir + '/' + file

            # load schem file
            schem = SchematicFile.load(file_path)

            # get block data, where each value corresponds to an index in the palette dictionary
            blockdata = schem.blocks.unpack()
            blockdata = blockdata.astype(object)

            # get the palette dictionary
            palette = schem.palette

            # reverse it so that the keys are indices and the values are the block names
            reverse_palette_dict = {y: x for x, y in palette.items()}

            # replace indices with their block names
            for key, value in reverse_palette_dict.items():
                blockdata[blockdata == key] = reverse_palette_dict[key]

            data_list.append(blockdata)

    print(len(data_list))
    combined = np.asarray(data_list)
    print(combined.shape)
    uniques, counts = np.unique(combined, return_counts=True)
    print(uniques)
    print(counts)
    np.save(schematic_dir + "/combined_blocknames.npy", combined)


create_combined_blockname_data('../ingame house schematics')
