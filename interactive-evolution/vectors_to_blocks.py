import grpc
import minecraft_pb2_grpc
from minecraft_pb2 import *
import numpy as np

channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

global DIRECTIONS, DIRECTIONS_2D, ORIENTATIONS, REMOVED_INDICES, RESTRICTED_BLOCKS

# CHOICE OF BLOCKS. 
REMOVED_INDICES = [0, 12, 50, 107, 111, 164, 182, 207, 215, 229, 234, 236, 247]
# THis will be taken into account in the case of restricted=True
RESTRICTED_BLOCKS = [QUARTZ_BLOCK, SLIME, WATER,REDSTONE_BLOCK, PISTON, STICKY_PISTON]

# PARAMETERS
N_BLOCK_TYPE = 254  # Number block type considered if not restricted blocks.

DIRECTIONS = ["N", "W", "S", "E", "U", "D"]
DIRECTIONS_2D = ["W", "E", "U", "D"]
ORIENTATIONS = ["O.N", "O.W", "O.S", "O.E", "O.U", "O.D"]

# The 4 is chosen for the structure to be over ground.
BOUNDS_WORLD = [[-30000000, 29999999],  [4, 255], [-30000000, 29999999]]


def symmetriseX(stuff):
    flip_stuff = np.flip(stuff, axis=0)
    symm_stuff = np.concatenate((flip_stuff, stuff), axis=0)
    return symm_stuff


# ############PRELIMINARIES####################+
def bound_coordinate(value, coord):
    """
    Restrict the coordinate to the bounds.
    INPUT:
        value: a value
        coord: the index of the coordinate (0,1,2)

    OUTPUT:
        the value bounded according the bounds registered in BOUNDS_MINECRAFT above.
    """
    low = BOUNDS_WORLD[coord][0]
    high = BOUNDS_WORLD[coord][1]
    return max(low, min(high, value))


def bounded(position):
    """
    Bounds the position according to BOUNDS_WORLD.
    INPUT:
        position: a 3D position.

    OUTPUT:
        bounded_position: a 3D position, within the boundaries given by BOUNDS_WORLD

    """
    bounded_position = [bound_coordinate(position[0], 0), bound_coordinate(
        position[1], 1), bound_coordinate(position[2], 2)]
    return bounded_position


def isBlock(index, oriented, graph, dimension):
    """
    Tells if a certain index is a block type index in the encoding used for RNN, depending if include orientations or not, and if use graph representation (with branching) or not.
    Input:
        index: integer
        oriented: boolean, indicate if the orientations are included in the encoding
        graph: boolean,  indicate if adopt a graph representation (with branching) or not

    Output:
        isBlock: boolean
    """
    if dimension == 2:
        dim = 4
    else:
        dim = 6
    isBlock = False
    if oriented and not graph:
        isBlock = bool(index > dim+7)
    elif not oriented and not graph:
        isBlock = bool(index > dim+1)
    elif oriented and graph:  # and graph, 2 more elements in encoding
        isBlock = bool(index > dim+9)
    elif not oriented and graph:
        isBlock = bool(index > dim+3)
    return isBlock


def allowed_blocks(restricted_flag):
    """
    Returns the list of allowed blocks.
    Input:
        restricted_flag: boolean, tells if restricted to a certain list of blocks        
    Output:
        ALLOWED_BLOCKS: list of block type (indices) allowed in the structure generation

    """
    if restricted_flag:
        ALLOWED_BLOCKS = RESTRICTED_BLOCKS
    else:  # all not removed
        ALLOWED_BLOCKS = []
        for i in range(N_BLOCK_TYPE):
            if not i in REMOVED_INDICES:
                ALLOWED_BLOCKS.append(i)
    return ALLOWED_BLOCKS


def move_coordinate(coord: (int, int, int), side_idx, delta=1):
    """
    Move coordinate along a certain direction.
    Input:
        coord: coordinates, integer 3-tuple.
        side_idx: denote a direction along which move, integer index, 0 is NORTH, etc. (cf. DIRECTIONS)
        *delta: an increment, integer

    Output:
        modified coordinate

    """
    """A quick way to increment a coordinate in the desired direction"""
    switcher = [  # directions=["north", "west", "south", "east", "up", "down"]
        lambda c: (c[0], c[1], c[2] - delta),  # Go North
        lambda c: (c[0] - delta, c[1], c[2]),  # Go West
        lambda c: (c[0], c[1], c[2] + delta),  # Go South
        lambda c: (c[0] + delta, c[1], c[2]),  # Go East
        lambda c: (c[0], c[1] + delta, c[2]),  # Go Up
        lambda c: (c[0], c[1] - delta, c[2]),  # Go Down
    ]
    return switcher[side_idx](coord)


def move_coordinate_2D(coord: (int, int, int), side_idx, delta=1):
    """
    Move coordinate along a certain direction, in 2D, along axis west-east and up-down only.
    Input:
        coord: coordinates, integer 3-tuple.
        side_idx: denote a direction along which move (here in 2D), integer index, 0 is WEST.
        *delta: an increment, integer

    Output:
        modified coordinate

    """
    """A quick way to increment a coordinate in the desired direction"""
    switcher = [  # directions_2D=["west", "east", "up", "down"]
        lambda c: (c[0] - delta, c[1], c[2]),  # Go West
        lambda c: (c[0] + delta, c[1], c[2]),  # Go East
        lambda c: (c[0], c[1] + delta, c[2]),  # Go Up
        lambda c: (c[0], c[1] - delta, c[2]),  # Go Down
    ]
    return switcher[side_idx](coord)

####################  PROCEDURE for ENCODING MLP   ###########

def build_zone(blocks, offset, restricted_flag, orientations, oriented):
    """
    Build a 3D structure, given by a tensor specifiying the value of each block type at each position (3D), and possibly orientations
    Inputs:
        blocks: np array size Mx*My*MZ, where Mx,My,Mz are bounds given as input.
        offset: position offset
        restricted_flag: boolean, tells if work with a restricted number of blocks
        orientations: np array size Mx*My*MZ, where Mx,My,Mz are bounds given as input.
        oriented: boolean indicating if shall take in account the orientations

    """
    positions = []
    blocks_index = []
    ALLOWED_BLOCKS = allowed_blocks(restricted_flag)  # Here want air
    orientations_ = []  # LIst with orientations
    for x in range(blocks.shape[0]):
        for y in range(blocks.shape[1]):  # this is height in minecraft
            for z in range(blocks.shape[2]):
                index = int(blocks[x, y, z])
                if not index == -1:  # AS INDEX - 1 means air block
                    try:
                        blocks_index.append(ALLOWED_BLOCKS[index])
                    except:
                        print(
                            "Following index out of bound of allowed blocks.", index)
                    # Update position
                    position = bounded([x+offset[0], y+offset[1], z+offset[2]])
                    positions.append(position)
                    if oriented:
                        orientations_.append(int(orientations[x, y, z]))

    #print("Building these block indices:", blocks_index)
    zone = [offset[0], offset[1], offset[2], offset[0]+blocks.shape[0],
            offset[1]+blocks.shape[1], offset[2]+blocks.shape[2]]
    if oriented:
        response = client.spawnBlocks(Blocks(blocks=[Block(position=Point(x=int(positions[i][0]), y=int(positions[i][1]), z=int(
            positions[i][2])), type=blocks_index[i], orientation=int(orientations_[i])) for i in range(len(blocks_index))]))
    else:
        response = client.spawnBlocks(Blocks(blocks=[Block(position=Point(x=int(positions[i][0]), y=int(positions[i][1]), z=int(
            positions[i][2])), type=blocks_index[i], orientation=NORTH) for i in range(len(blocks_index))]))



####################CLEANING PROCEDURES ###########


def clean_positions(positions):
    """
    As a way to clear out a space, place a block of AIR in each of the indicated positions.
    Input:
        positions: np.array of size N*3
    """
    for i in range(positions.shape[0]):
        response = client.spawnBlocks(Blocks(blocks=[Block(position=Point(x=int(positions[i, 0]), y=int(
            positions[i, 1]), z=int(positions[i, 2])), type=AIR, orientation=NORTH)]))
        # print(response)


def clean_batch(positions_batch):
    """
    Cleans a list of positions, by replacing them with air.
    Input:
        positions_batch: list of positions (np.array)

    """
    for positions in positions_batch:
        if len(positions) > 0:
            clean_positions(positions)


def clean_zone(bounds, offset):
    """
    Cleans an area of space within certain bounds, by replacing them by block of AIR.
    Input:
        bounds: dimensions of the zone, list of 3 elements.
        offset: offset position.

    """
    zone = [offset[0], 4, offset[2], offset[0] +
            bounds[0], 4+bounds[1], offset[2]+bounds[2]]
    print("Cleaning the following zone:", zone)
    response = client.fillCube(FillCubeRequest(
        cube=Cube(min=Point(x=int(offset[0]-10), y=int(4), z=int(offset[2]-10)), max=Point(x=int(offset[0]+bounds[0]+10), y=int(
            4+bounds[1]+10), z=int(offset[2]+bounds[2]+10))),  
        type=AIR
    ))
    print(response)



def convert_to_block_type(index, restricted_flag):
    """
    Given an index between 0 to N_VALID_BLOCK_TYPE
    Return the corresponding index between 0 to N_BLOCK_TYPE
    """
    ALLOWED_BLOCKS = allowed_blocks(restricted_flag, True)
    block_type = ALLOWED_BLOCKS[index]
    return block_type


def build_one_from_one_hot(block, pos):
    """
    Input:
         block: one hot encoding of a block type. numpy array size M, where M is number different blocks possible. 
         pos: one 3D position for a block, numpy array size 3
    Output:
         Build a block, of the type given, with the position given

    """
    index = np.argmax(
        block)  # Use Argmax even though it may be faster as know only one element non zero.
    block_type = convert_to_block_type(index)
    print("Current Block type", block_type)
    # PLACE A BLOCK
    response = client.spawnBlocks(Blocks(blocks=[Block(position=Point(
        x=pos[0], y=pos[1], z=pos[2]), type=block_type, orientation=NORTH)]))


def build_from_one_hot(blocks, pos):
    """
    Input:
        blocks: one hot encoding of several blocks; i.e. vector size N*M, where M is number different blocks possible, N is number blocks to place. 
        pos: 3D positions of these blocks; ; i.e. vector size N*3, where N is number blocks to place.
    Output:
        Build some block, of the types given, with the positions given.

    """
    assert blocks.shape[0] == pos.shape[0]
    indices = np.argmax(blocks, axis=1)
    print("Building these block indices:", indices)
    response = client.spawnBlocks(Blocks(blocks=[Block(position=Point(x=pos[i, 0], y=pos[i, 1], z=pos[i, 2]), type=convert_to_block_type(
        indices[i]), orientation=NORTH) for i in range(blocks.shape[0])]))

