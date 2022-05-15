import time
import argparse
import sys
from os.path import join, exists
from os import mkdir
import numpy as np

from evolution_strategy_static import EvolutionStrategyStatic
from policies import VAE
from vectors_to_blocks import RESTRICTED_BLOCKS, REMOVED_INDICES

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--choice_batch', type=int, default=2, metavar='',
                        help='Number of structures among which to choose one.')
    parser.add_argument('--oriented', type=int, default=1, metavar='',
                        help='0 or 1. Indicate if shall incorporate the orientations in the encoding.')
    parser.add_argument('--position', type=list, default=[0, 10, 0], metavar='',
                        help='Initial position for player advised, around which the structures will be evolved.')
    parser.add_argument('--lr', type=float,  default=0.1,
                        metavar='', help='ES learning rate.')
    parser.add_argument('--decay', type=float,  default=0.99, metavar='',
                        help='ES and learning rate decay.') 
    parser.add_argument('--sigma', type=float,  default=0.4, metavar='',
                        help='ES sigma: modulates the amount of noise used to populate each new generation, the higher the more the entities will vary')
    parser.add_argument('--generations', type=int, default=30,
                        metavar='', help='Number of generations that the ES will run.')
    parser.add_argument('--population_size', type=int, default=2, metavar='',
                        help='Size of population (needs to be pair and be a multiple of choice_batch or will be approximated).')
    parser.add_argument('--top_k', type=int, default=1, metavar='',
                        help='Top-k sampling, for a stochastic generation of structures. For the deterministic case, choose k=1.')
    parser.add_argument('--folder', type=str, default='weights',
                        metavar='', help='folder to store the evolved weights ')


    args = parser.parse_args()

    assert args.dimension == 2 or args.dimension == 3
    assert args.choice_batch <= args.population_size

    if not exists(args.folder):
        mkdir(args.folder)

    # Initialise generator network and create constructor dictionary

    if args.generator == 'VAE':
        generator_init_params = {
            'output_dim': int(len(RESTRICTED_BLOCKS) + 6*args.oriented+1),
            'embedding_dim': 50,
            # Bounds within which will query the network to build a structure. If 2D, the 3rd dimension will be ignored.
            'bounds': [6, 10, 10],
            # If query blocks only within a radial bound.
            'radial_bound': True,
            # If radial_bound=true, will inquire only blocks within a certain radius. Has to be between 0 and 1. Radius computed then depending on above bounds.
            'max_radius': 0.9,
            'top_k': args.top_k,
            'min_size': 3,  # Minimum number of block for a structure to proposed under the human rating
            'position': args.position,
            'choice_batch': args.choice_batch,
            'population_size': args.population_size,
            'oriented': bool(args.oriented),
        }
        p = VAE

    else:
        raise NotImplementedError

    # Initialise the EvolutionStrategy class
    print('\nInitilisating ES for ' + args.generator)
    es = EvolutionStrategyStatic(p.get_weights(), generator_init_params=generator_init_params,
                                sigma=args.sigma, learning_rate=args.lr, decay=args.decay)

    # Start the evolution
    es.run(args.generations, path=args.folder)


if __name__ == '__main__':
    main(sys.argv)
