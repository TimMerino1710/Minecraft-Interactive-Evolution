import numpy as np
import random

data = np.load('H:\\joined_cureated_stretched_rotated_categorical_onehot.npy')
data = np.argmax(data, axis=4)
data[data != 0] = 1

for house in data[:10]:
    num_delete = random.randint(10, 30)

    # zip into list of (x, y, z) index tuples
    x, y, z = np.nonzero(house)
    rm_idxs = np.random.choice(len(x), num_delete, replace=False)
    print("num nonzero: ", np.count_nonzero(house))
    print("removing ", num_delete)
    for idx in rm_idxs:
        house[x[idx], y[idx], z[idx]] = 0
    print("num nonzero after: ", np.count_nonzero(house))