import numpy as np
import os
import sys

if __name__=="__main__":
    rootdir = sys.argv[1]
    outdir = sys.argv[2]
    count = 1
    for dirname, subdirlist, filelist in os.walk(rootdir):
        for file in filelist:
            if file.startswith('test_decoded') and file.endswith('.npy'):
                orig = np.load(dirname + file)
                orig[np.where(orig >= .25)] = 1
                orig[np.where(orig < .25)] = 0
                np.save(outdir + os.path.splitext(file)[0] + "_rounded", orig)
