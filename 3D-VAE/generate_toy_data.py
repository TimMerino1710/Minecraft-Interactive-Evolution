import numpy as np
from nbtschematic import SchematicFile
data = np.zeros((1000, 32, 32, 32))
data[:, :16, :16, :16] = 1
np.save('../house_combined_numpy_file/toy_data.npy', data)
# for i in range(2):
#
#     sample = data[i]
#     arr_axes = (sample.shape[0], sample.shape[1], sample.shape[2])
#     sf = SchematicFile(shape=arr_axes)
#     assert sf.blocks.shape == arr_axes
#     for index, block_id in np.ndenumerate(sample):
#         sf.blocks[index[0], index[1], index[2]] = block_id
#     sf.save(str(i) + ".schematic")