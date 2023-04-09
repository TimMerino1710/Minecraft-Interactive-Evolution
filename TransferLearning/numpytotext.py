import os
import numpy as np
import json

def convert_samples_to_txt(samples_file_path):
    filename = os.path.splitext(os.path.basename(samples_file_path))[0]
    out_dir = os.path.dirname(samples_file_path) + "/" + filename + "_text"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    data = np.load(samples_file_path, allow_pickle=True)
    for i, struct in enumerate(data):
        struct = struct.squeeze()
        print(struct.shape)
        txt_filename = filename + "_" + str(i) + ".txt"
        txt_path = os.path.join(out_dir, txt_filename)
        json.dump(struct.astype(int).tolist(), open(txt_path, 'w+'))


convert_samples_to_txt('C:/Users/timme/Documents/MinecraftStructurePCG/TransferLearning/GAN_generated_samples_Shapenet/minecraftGAN_1/sample_epoch_3000.npy') 
