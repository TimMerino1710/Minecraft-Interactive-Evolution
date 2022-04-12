#!/usr/bin/env python3

import numpy as np
from nbtschematic import SchematicFile
import os
import sys

if __name__=="__main__":
    rootdir = sys.argv[1]
    count = 1
    for dirname, subdirlist, filelist in os.walk(rootdir):
        if "schematic.npy" in filelist:
            data = np.load(dirname+r"/schematic.npy")[:,:,:,0]
            arr_axes = (data.shape[0], data.shape[1], data.shape[2])
            sf = SchematicFile(shape=arr_axes)
            assert sf.blocks.shape == arr_axes
            for index, block_id in np.ndenumerate(data):
                sf.blocks[index[0], index[1], index[2]] = block_id
            sf.save("3DCraft" + str(count) + ".schematic")
            count += 1
