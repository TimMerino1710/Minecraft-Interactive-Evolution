import tensorflow
import math
import numpy as np
from vectors_to_blocks import isBlock, symmetriseX


class VAE:

    def get_weights(self):
        param = nn.utils.parameters_to_vector(
            self.parameters()).detach().numpy()
        print("weights param size", param.shape)
        return param

    def generate_structure(self, generator_init_params):
        """
        Given a policy network, and certain parameters, generate a 3D structure with top-k-sampling.
        Outputs:
            blocks: int-np-array of size Mx*My*Mz, where M are the bounds provided as argument of train.
                blocks[x,y,z] is an index indicating which block type shall be found there. If -1, means it should be air.
            orientations: int-np-array of size Mx*My*Mz, where M are the bounds provided as argument of train.
                orientations[x,y,z] is an index between 0 to 5 indicating which orientations shall the block be oriented. 
                Only matter if take in account the orientations.

        """
        bounds = generator_init_params['bounds']
        oriented = generator_init_params['oriented']
        top_k = max(1, generator_init_params['top_k'])
        # indicate if build AIR block or not
        density_threshold = generator_init_params['density_threshold']
        # BUILD CREATURE look at all blocks within bounds
        # BY DEFAULT BLOCK TYPES ARE AIR, so EMPTY
        blocks = (-1) * np.ones((bounds[0], bounds[1], bounds[2]), dtype=int)
        orientations = np.zeros((bounds[0], bounds[1], bounds[2]), dtype=int)
        input_symmetry = generator_init_params['input_symmetry']
        max_radius = generator_init_params['max_radius']
        radial_bound = generator_init_params['radial_bound']
        matter_blocks = 0

        # generate
        
        return blocks, orientations, matter_blocks

    