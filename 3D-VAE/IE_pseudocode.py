from keras.models import load_model
import numpy as np
from random import random

compression_list = [[0, 6, 26, 27, 28, 30, 55, 63, 65, 66, 68, 69, 70, 72, 77, 97, 104, 117, 127, 131, 132, 143, 144, 147, 148, 149, 167, 171, 50, 51, 76, 105, 123],[1, 4, 7, 29, 33, 34, 46, 48, 49, 52, 54, 61, 87, 89, 98, 45, 112, 120, 121, 139, 155, 158, 168, 169, 201, 202, 206, 215, 216, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 35, 14, 15, 16, 21, 56, 73, 129, 153],[2, 3, 82, 88, 159, 172, 173, 214],[5, 17, 25, 47, 58, 84, 96, 116, 130, 140, 146, 151, 154, 162],[8, 9, 10, 11, 213],[12, 13, 19, 24, 179],[18, 31, 32, 81, 86, 91, 103, 106, 161, 170, 199, 200, 207],[20, 92, 102, 160, 95],[22, 23, 41, 42, 57, 118, 133, 138, 145, 152, 165],[37, 38, 39, 40, 59, 83, 110, 115, 141, 142, 175],[43, 44, 92, 125, 126, 181, 182, 204, 205],[53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164, 180, 203],[64, 71, 193, 194, 195, 196, 197],[78, 79, 80, 174],[85, 101, 107, 113, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 198],]


## Take command line arguements to specify whether we're loading a GAN or vae
model_type = 'gan'

gan_model_path = 'example/example/gan_generator.h5'
vae_model_path = 'example/example/vae_decoderr.h5'

pop_size = 5

mutation_prob = .05
mutation_amt = .1

# TODO: get the size of the z dimension by reading the first layer of the model
z_dim = 200

# generates num_pop number of random normal vectors with size z_dim
def create_starting_population_gan(num_pop):
    return np.random.normal(0, 1, (num_pop, z_dim))

# turns integer values back into block IDs by inverting our compression
def decompress(data):
    for i in range(len(compression_list)):
        data[data == i] = compression_list[i][0]
    return data

# take latent vectors, and generate structures from them
def gan_generate_from_latent(gan_model, latent_vectors):
    generated_structures = gan_model.predict(latent_vectors)

    # a categorical GAN will output one-hot encoded, so this must turned back into categorical
    generated_structures = np.argmax(generated_structures, axis=4)

    # this results in integer values between 0 and 14. These must be decompressed back into their actual minecraft blockids to render properly
    generated_structures = decompress(generated_structures)

    return generated_structures


# renders our population by spawning them in on the evocraft server
# TODO: how do we render it? for now, just render them in a single row, so it's easy to see which index aligns with which structure
def render_population(model, population):
    generated_structures = gan_generate_from_latent(model, population)
    for struc in population:
        # TODO: use evocraft to draw all these into the server.
        pass

# TODO: gets users selection as an integer keyboard input
def get_user_input():
    return 5


# mutates a single vector
def mutate(vector):
    # iterate through each value in the vector
    for i in range(len(vector)):
        # change each value by adding or subtracting mutation_amt with probability mutation_prob
        if random.uniform(0, 1) <= mutation_prob:
            if random.uniform(0, 1) <= .5:
                vector[i] = vector[i] - mutation_amt
            else:
                vector[i] = vector[i] + mutation_amt




# creates a population based on the selected latent vector.
def generate_mutated_population(selected_latent):
    new_pop = selected_latent
    # generate pop_size - 1 number of new latents vectors, because our selected latent will be added to the next population
    for i in range(pop_size - 1):
        new_latent_vector = np.copy(selected_latent)
        new_latent_vector = mutate(new_latent_vector)

        # TODO: figure out how to stack this. We want to end up with a size of (num_pop, z_dim)
        new_pop = np.stack(new_pop, new_latent_vector)
    return new_pop



def train():
    ## Two seaparate code blocks depending on which model
    if model_type == 'gan':
        model = load_model(gan_model_path)

        # create a starting population as a list of latent vectors
        population = create_starting_population_gan(pop_size)

        # begin interactive evolution loop
        while True:
            # render the population
            render_population(model, population)

            # TODO: get user input, which represents the index of the structure they select to further evolve
            selection = get_user_input()

            # get our latent vector to be mutated
            selected_latent = population[selection]

            population = generate_mutated_population(selected_latent)


    else:
        #TODO: do the same for a VAE
        pass

