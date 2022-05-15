import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from nbtschematic import SchematicFile
from pprint import pprint
import json
import tensorflow as tf

HOUSE_NUMPY_DIR = '../house_numpy_files'
HOUSE_OUT_DIR = '../house_combined_numpy_file'
SANITY_CHECK_DIR = 'sanity_checks'
COMBINED_FILE = '../house_combined_numpy_file/combined.npy'
blockid_dict = {
0:  "Air",
1:  "Stone",
2:  "Grass",
3:  "Dirt",
4:  "Cobblestone",
5:  "Oak Wood Plank",
6:  "Oak Sapling",
7:  "Bedrock",
8:  "Flowing Water",
9:  "Still Water",
10:  "Flowing Lava",
11:  "Still Lava",
12:  "Sand",
13:  "Gravel",
14:  "Gold Ore",
15:  "Iron Ore",
16:  "Coal Ore",
17:  "Oak Wood",
18:  "Oak Leaves",
19:  "Sponge",
20:  "Glass",
21:  "Lapis Lazuli Ore",
22:  "Lapis Lazuli Block",
23:  "Dispenser",
24:  "Sandstone",
25:  "Note Block",
26:  "Bed",
27:  "Powered Rail",
28:  "Detector Rail",
29:  "Sticky Piston",
30:  "Cobweb",
31:  "Dead Shrub",
32:  "Dead Bush",
33:  "Piston",
34:  "Piston Head",
35:  "White Wool",
37:  "Dandelion",
38:  "Poppy",
39:  "Brown Mushroom",
40:  "Red Mushroom",
41:  "Gold Block",
42:  "Iron Block",
43:  "Double Stone Slab",
44:  "Stone Slab",
45:  "Bricks",
46:  "TNT",
47:  "Bookshelf",
48:  "Moss Stone",
49:  "Obsidian",
50:  "Torch",
51:  "Fire",
52:  "Monster Spawner",
53:  "Oak Wood Stairs",
54:  "Chest",
55:  "Redstone Wire",
56:  "Diamond Ore",
57:  "Diamond Block",
58:  "Crafting Table",
59:  "Wheat Crops",
60:  "Farmland",
61:  "Furnace",
62:  "Burning Furnace",
63:  "Standing Sign Block",
64:  "Oak Door Block",
65:  "Ladder",
66:  "Rail",
67:  "Cobblestone Stairs",
68:  "Wall-mounted Sign Block",
69:  "Lever",
70:  "Stone Pressure Plate",
71:  "Iron Door Block",
72:  "Wooden Pressure Plate",
73:  "Redstone Ore",
74:  "Glowing Redstone Ore",
75:  "Redstone Torch (off)",
76:  "Redstone Torch (on)",
77:  "Stone Button",
78:  "Snow",
79:  "Ice",
80:  "Snow Block",
81:  "Cactus",
82:  "Clay",
83:  "Sugar Canes",
84:  "Jukebox",
85:  "Oak Fence",
86:  "Pumpkin",
87:  "Netherrack",
88:  "Soul Sand",
89:  "Glowstone",
90:  "Nether Portal",
91:  "Jack o'Lantern",
92:  "Cake Block",
93:  "Redstone Repeater Block (off)",
94:  "Redstone Repeater Block (on)",
95:  "White Stained Glass",
96:  "Wooden Trapdoor",
97:  "Stone Monster Egg",
98:  "Stone Bricks",
99:  "Brown Mushroom Block",
100:  "Red Mushroom Block",
101:  "Iron Bars",
102:  "Glass Pane",
103:  "Melon Block",
104:  "Pumpkin Stem",
105:  "Melon Stem",
106:  "Vines",
107:  "Oak Fence Gate",
108:  "Brick Stairs",
109:  "Stone Brick Stairs",
110:  "Mycelium",
111:  "Lily Pad",
112:  "Nether Brick",
113:  "Nether Brick Fence",
114:  "Nether Brick Stairs",
115:  "Nether Wart",
116:  "Enchantment Table",
117:  "Brewing Stand",
118:  "Cauldron",
119:  "End Portal",
120:  "End Portal Frame",
121:  "End Stone",
122:  "Dragon Egg",
123:  "Redstone Lamp (inactive)",
124:  "Redstone Lamp (active)",
125:  "Double Oak Wood Slab",
126:  "Oak Wood Slab",
127:  "Cocoa",
128:  "Sandstone Stairs",
129:  "Emerald Ore",
130:  "Ender Chest",
131:  "Tripwire Hook",
132:  "Tripwire",
133:  "Emerald Block",
134:  "Spruce Wood Stairs",
135:  "Birch Wood Stairs",
136:  "Jungle Wood Stairs",
137:  "Command Block",
138:  "Beacon",
139:  "Cobblestone Wall",
140:  "Flower Pot",
141:  "Carrots",
142:  "Potatoes",
143:  "Wooden Button",
144:  "Mob Head",
145:  "Anvil",
146:  "Trapped Chest",
147:  "Weighted Pressure Plate (light)",
148:  "Weighted Pressure Plate (heavy)",
149:  "Redstone Comparator (inactive)",
150:  "Redstone Comparator (active)",
151:  "Daylight Sensor",
152:  "Redstone Block",
153:  "Nether Quartz Ore",
154:  "Hopper",
155:  "Quartz Block",
156:  "Quartz Stairs",
157:  "Activator Rail",
158:  "Dropper",
159:  "White Hardened Clay",
160:  "White Stained Glass Pane",
161:  "Acacia Leaves",
162:  "Acacia Wood",
163:  "Acacia Wood Stairs",
164:  "Dark Oak Wood Stairs",
165:  "Slime Block",
166:  "Barrier",
167:  "Iron Trapdoor",
168:  "Prismarine",
169:  "Sea Lantern",
170:  "Hay Bale",
171:  "White Carpet",
172:  "Hardened Clay",
173:  "Block of Coal",
174:  "Packed Ice",
175:  "Sunflower",
176:  "Free-standing Banner",
177:  "Wall-mounted Banner",
178:  "Inverted Daylight Sensor",
179:  "Red Sandstone",
180:  "Red Sandstone Stairs",
181:  "Double Red Sandstone Slab",
182:  "Red Sandstone Slab",
183:  "Spruce Fence Gate",
184:  "Birch Fence Gate",
185:  "Jungle Fence Gate",
186:  "Dark Oak Fence Gate",
187:  "Acacia Fence Gate",
188:  "Spruce Fence",
189:  "Birch Fence",
190:  "Jungle Fence",
191:  "Dark Oak Fence",
192:  "Acacia Fence",
193:  "Spruce Door Block",
194:  "Birch Door Block",
195:  "Jungle Door Block",
196:  "Acacia Door Block",
197:  "Dark Oak Door Block",
198:  "End Rod",
199:  "Chorus Plant",
200:  "Chorus Flower",
201:  "Purpur Block",
202:  "Purpur Pillar",
203:  "Purpur Stairs",
204:  "Purpur Double Slab",
205:  "Purpur Slab",
206:  "End Stone Bricks",
207:  "Beetroot Block",
208:  "Grass Path",
209:  "End Gateway",
210:  "Repeating Command Block",
211:  "Chain Command Block",
212:  "Frosted Ice",
213:  "Magma Block",
214:  "Nether Wart Block",
215:  "Red Nether Brick",
216:  "Bone Block",
217:  "Structure Void",
218:  "Observer",
219:  "White Shulker Box",
220:  "Orange Shulker Box",
221:  "Magenta Shulker Box",
222:  "Light Blue Shulker Box",
223:  "Yellow Shulker Box",
224:  "Lime Shulker Box",
225:  "Pink Shulker Box",
226:  "Gray Shulker Box",
227:  "Light Gray Shulker Box",
228:  "Cyan Shulker Box",
229:  "Purple Shulker Box",
230:  "Blue Shulker Box",
231:  "Brown Shulker Box",
232:  "Green Shulker Box",
233:  "Red Shulker Box",
234:  "Black Shulker Box",
235:  "White Glazed Terracotta",
236:  "Orange Glazed Terracotta",
237:  "Magenta Glazed Terracotta",
238:  "Light Blue Glazed Terracotta",
239:  "Yellow Glazed Terracotta",
240:  "Lime Glazed Terracotta",
241:  "Pink Glazed Terracotta",
242:  "Gray Glazed Terracotta",
243:  "Light Gray Glazed Terracotta",
244:  "Cyan Glazed Terracotta",
245:  "Purple Glazed Terracotta",
246:  "Blue Glazed Terracotta",
247:  "Brown Glazed Terracotta",
248:  "Green Glazed Terracotta",
249:  "Red Glazed Terracotta",
250:  "Black Glazed Terracotta",
251:  "White Concrete",
252:  "White Concrete Powder",
255:  "Structure Block",
256:  "Iron Shovel",
257:  "Iron Pickaxe",
258:  "Iron Axe",
259:  "Flint and Steel",
260:  "Apple",
261:  "Bow",
262:  "Arrow",
263:  "Coal",
264:  "Diamond",
265:  "Iron Ingot",
266:  "Gold Ingot",
267:  "Iron Sword",
268:  "Wooden Sword",
269:  "Wooden Shovel",
270:  "Wooden Pickaxe",
271:  "Wooden Axe",
272:  "Stone Sword",
273:  "Stone Shovel",
274:  "Stone Pickaxe",
275:  "Stone Axe",
276:  "Diamond Sword",
277:  "Diamond Shovel",
278:  "Diamond Pickaxe",
279:  "Diamond Axe",
280:  "Stick",
281:  "Bowl",
282:  "Mushroom Stew",
283:  "Golden Sword",
284:  "Golden Shovel",
285:  "Golden Pickaxe",
286:  "Golden Axe",
287:  "String",
288:  "Feather",
289:  "Gunpowder",
290:  "Wooden Hoe",
291:  "Stone Hoe",
292:  "Iron Hoe",
293:  "Diamond Hoe",
294:  "Golden Hoe",
295:  "Wheat Seeds",
296:  "Wheat",
297:  "Bread",
298:  "Leather Helmet",
299:  "Leather Tunic",
300:  "Leather Pants",
301:  "Leather Boots",
302:  "Chainmail Helmet",
303:  "Chainmail Chestplate",
304:  "Chainmail Leggings",
305:  "Chainmail Boots",
306:  "Iron Helmet",
307:  "Iron Chestplate",
308:  "Iron Leggings",
309:  "Iron Boots",
310:  "Diamond Helmet",
311:  "Diamond Chestplate",
312:  "Diamond Leggings",
313:  "Diamond Boots",
314:  "Golden Helmet",
315:  "Golden Chestplate",
316:  "Golden Leggings",
317:  "Golden Boots",
318:  "Flint",
319:  "Raw Porkchop",
320:  "Cooked Porkchop",
321:  "Painting",
322:  "Golden Apple",
323:  "Sign",
324:  "Oak Door",
325:  "Bucket",
326:  "Water Bucket",
327:  "Lava Bucket",
328:  "Minecart",
329:  "Saddle",
330:  "Iron Door",
331:  "Redstone",
332:  "Snowball",
333:  "Oak Boat",
334:  "Leather",
335:  "Milk Bucket",
336:  "Brick",
337:  "Clay",
338:  "Sugar Canes",
339:  "Paper",
340:  "Book",
341:  "Slimeball",
342:  "Minecart with Chest",
343:  "Minecart with Furnace",
344:  "Egg",
345:  "Compass",
346:  "Fishing Rod",
347:  "Clock",
348:  "Glowstone Dust",
349:  "Raw Fish",
350:  "Cooked Fish",
351:  "Ink Sack",
352:  "Bone",
353:  "Sugar",
354:  "Cake",
355:  "Bed",
356:  "Redstone Repeater",
357:  "Cookie",
358:  "Map",
359:  "Shears",
360:  "Melon",
361:  "Pumpkin Seeds",
362:  "Melon Seeds",
363:  "Raw Beef",
364:  "Steak",
365:  "Raw Chicken",
366:  "Cooked Chicken",
367:  "Rotten Flesh",
368:  "Ender Pearl",
369:  "Blaze Rod",
370:  "Ghast Tear",
371:  "Gold Nugget",
372:  "Nether Wart",
373:  "Potion",
374:  "Glass Bottle",
375:  "Spider Eye",
376:  "Fermented Spider Eye",
377:  "Blaze Powder",
378:  "Magma Cream",
379:  "Brewing Stand",
380:  "Cauldron",
381:  "Eye of Ender",
382:  "Glistering Melon",
384:  "Bottle o' Enchanting",
385:  "Fire Charge",
386:  "Book and Quill",
387:  "Written Book",
388:  "Emerald",
389:  "Item Frame",
390:  "Flower Pot",
391:  "Carrot",
392:  "Potato",
393:  "Baked Potato",
394:  "Poisonous Potato",
395:  "Empty Map",
396:  "Golden Carrot",
397:  "Mob Head (Skeleton)",
398:  "Carrot on a Stick",
399:  "Nether Star",
400:  "Pumpkin Pie",
401:  "Firework Rocket",
402:  "Firework Star",
403:  "Enchanted Book",
404:  "Redstone Comparator",
405:  "Nether Brick",
406:  "Nether Quartz",
407:  "Minecart with TNT",
408:  "Minecart with Hopper",
409:  "Prismarine Shard",
410:  "Prismarine Crystals",
411:  "Raw Rabbit",
412:  "Cooked Rabbit",
413:  "Rabbit Stew",
414:  "Rabbit's Foot",
415:  "Rabbit Hide",
416:  "Armor Stand",
417:  "Iron Horse Armor",
418:  "Golden Horse Armor",
419:  "Diamond Horse Armor",
420:  "Lead",
421:  "Name Tag",
422:  "Minecart with Command Block",
423:  "Raw Mutton",
424:  "Cooked Mutton",
425:  "Banner",
426:  "End Crystal",
427:  "Spruce Door",
428:  "Birch Door",
429:  "Jungle Door",
430:  "Acacia Door",
431:  "Dark Oak Door",
432:  "Chorus Fruit",
433:  "Popped Chorus Fruit",
434:  "Beetroot",
435:  "Beetroot Seeds",
436:  "Beetroot Soup",
437:  "Dragon's Breath",
438:  "Splash Potion",
439:  "Spectral Arrow",
440:  "Tipped Arrow",
441:  "Lingering Potion",
442:  "Shield",
443:  "Elytra",
444:  "Spruce Boat",
445:  "Birch Boat",
446:  "Jungle Boat",
447:  "Acacia Boat",
448:  "Dark Oak Boat",
449:  "Totem of Undying",
450:  "Shulker Shell",
452:  "Iron Nugget",
453:  "Knowledge Book",
2256:  "13 Disc",
2257:  "Cat Disc",
2258:  "Blocks Disc",
2259:  "Chirp Disc",
2260:  "Far Disc",
2261:  "Mall Disc",
2262:  "Mellohi Disc",
2263:  "Stal Disc",
2264:  "Strad Disc",
2265:  "Ward Disc",
2266:  "11 Disc",
2267:  "Wait Disc",
}
# a list, where each element is a sublist of all blockids corresponding to a category. Anything present in a sublist can be replaced by the first elemnt of the list (the most general element)
compression_list = [[0, 6, 26, 27, 28, 30, 55, 63, 65, 66, 68, 69, 70, 72, 77, 97, 104, 117, 127, 131, 132, 143, 144, 147, 148, 149, 167, 171, 50, 51, 76, 105, 123],[1, 4, 7, 29, 33, 34, 46, 48, 49, 52, 54, 61, 87, 89, 98, 45, 112, 120, 121, 139, 155, 158, 168, 169, 201, 202, 206, 215, 216, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 35, 14, 15, 16, 21, 56, 73, 129, 153],[2, 3, 82, 88, 159, 172, 173, 214],[5, 17, 25, 47, 58, 84, 96, 116, 130, 140, 146, 151, 154, 162],[8, 9, 10, 11, 213],[12, 13, 19, 24, 179],[18, 31, 32, 81, 86, 91, 103, 106, 161, 170, 199, 200, 207],[20, 92, 102, 160, 95],[22, 23, 41, 42, 57, 118, 133, 138, 145, 152, 165],[37, 38, 39, 40, 59, 83, 110, 115, 141, 142, 175],[43, 44, 92, 125, 126, 181, 182, 204, 205],[53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164, 180, 203],[64, 71, 193, 194, 195, 196, 197],[78, 79, 80, 174],[85, 101, 107, 113, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 198],]


# loads all 3d craft npy files into a list of np arrays
def load_data():
    data = []
    for file in os.listdir(HOUSE_NUMPY_DIR):
        if file.endswith('.npy'):
            data.append(np.load(HOUSE_NUMPY_DIR + '/' + file)[:, :, :, 0])
    return data

# takes a list of np arrays, and calculates the largest size in x, y, z across all entries
def calc_max_axis(data_list):
    max_x = 0
    max_y = 0
    max_z = 0
    for array in data_list:
        shape = array.shape
        x_len, y_len, z_len = shape[0], shape[1], shape[2]
        if x_len > max_x:
            max_x = x_len
        if y_len > max_y:
            max_y = y_len
        if z_len > max_z:
            max_z = z_len
    return max_x, max_y, max_z

# create dataframe with shapes of each
def get_build_dim_df(data_list):
    df = pd.DataFrame(columns = ['x_len', 'y_len', 'z_len'])
    for i, array in enumerate(data_list):
        dim_list = list(array.shape)
        df.loc[i] = dim_list
    df = df.astype(int)
    return df

# returns a a filtered data_list where all samples fit within specified size of (x_max, y_max, z_max)
def cut_to_dim(dim_df, data_list, x_max, y_max, z_max):
    filtered_idx = dim_df.index[(dim_df['x_len'] <= x_max) & (dim_df['y_len']<= y_max) & (dim_df['z_len'] < z_max)].tolist()
    return [data_list[i] for i in filtered_idx]

# write a np array, of size [# samples, xdim, ydim, zdim] to a file for easier loading
def write_to_npy_file(data_list, file_name, out_dir=HOUSE_OUT_DIR):
    data_np_array = np.asarray(data_list)
    np.save(out_dir + "/" + file_name, data_np_array)

# combines y structures into a (y, dimx, dimy, dimz) arrayy
def combine_data(data_list):
    return np.asarray(data_list)

# pads all arrays in data_list to be size (max_x, max_y, max_z) by padding edges with 0s
def pad_arrays(data_list, max_x, max_y, max_z):
    padded_data_list = []
    for array in data_list:
        shape = array.shape
        cur_x, cur_y, cur_z = shape[0], shape[1], shape[2]
        padded = np.pad(array, ((0, max_x - cur_x), (0, max_y - cur_y), (0, max_z - cur_z)), mode='constant')
        padded_data_list.append(padded)
    return padded_data_list

# turns every block into stone, for testing with binary voxel models
def convert_to_stone_only(data_list):
    # for array in data_list:
    for array in data_list:
        array[np.where(array != 0)] = 1
    return data_list

# convert npy files in dir to schematic files
def convert_to_schem(dir):
    count = 1
    for dirname, _, filelist in os.walk(dir):
        for file in filelist:
            if file.endswith('.npy'):
                print(file)
                data = np.load(dirname + "/" + file)
                arr_axes = (data.shape[0], data.shape[1], data.shape[2])
                sf = SchematicFile(shape=arr_axes)
                assert sf.blocks.shape == arr_axes
                for index, block_id in np.ndenumerate(data):
                    sf.blocks[index[0], index[1], index[2]] = block_id
                sf.save(dir + "/" + file + ".schematic")
                count += 1

# debug method to output an original array as a npy file, as well as a list of transformed arrays, to confirm that any augmentations / manipulations are working correctly
def sanity_check(array, others):
    np.save(SANITY_CHECK_DIR + "/before.npy", array)
    for i, array in enumerate(others):
        np.save(SANITY_CHECK_DIR + "/after" + str(i) + ".npy", array)


# from paper:
# "We change the binary voxel range from {0,1} to {-1,5} to encourage the network to pay more attention to positive entries"
# Doesn't play nice without doing some custom work with loss functions which is beyond me, don't recommend using
def scale_data(data):
    scaled_data = []
    for array in data:
        new = array.astype(int)
        new[new == 0] = -1
        new[new == 1] = 5
        scaled_data.append(new)
    return scaled_data

# rotates samples around an axis (the Z axis? not sure, minecraft axes are confusing, but it rotates in the way you would expect)
def rotation_augmentation(data):
    rotated_data = []
    for array in data:
        rotated_90 = np.rot90(array, axes=(1, 2))
        rotated_180 = np.rot90(rotated_90, axes=(1, 2))
        rotated_270 = np.rot90(rotated_180, axes=(1, 2))
        rotated_data.append(array)
        rotated_data.append(rotated_90)
        rotated_data.append(rotated_180)
        rotated_data.append(rotated_270)
    return rotated_data

#TODO:
# reflect each sample around the X and/or Y axis
def reflection_augmentation(data):
    flipped_data = []
    for array in data:
        copy = np.copy(array)
        flipped_X = np.flip(copy, 2)
        flipped_data.append(copy)
        flipped_data.append(flipped_X)
    return flipped_data
#TODO:
# the trickiest augmentation, translate our samples in the X and y dimensions as much as possible
# this will require computing how much we can translate (how much empty space we have on each side, or the buffer)
# decisions about "buffer": do we take the minimum buffer and use that to limit our translation? or do we translate each sample as much as possible within its buffer (will underrepresent larger builds, overrepresent smaller)?
# alternatively, we can increase our maximum size by padding with empty space, that way everything gets equal translations. Will increase dimensionality
# what step size do we take? 1 block? 3? 5?
def translation_augmentation(data):
    pass

#TODO:
# center data. This will take every array, find its corners(?) find a midpoint, and center that in the 3d array if possible
# will require an algorithm for finding those corners, calculating a midpoint, compensating for outliers, compensating for any possible overflow issues
def center_data():
    pass

# returns the unique block IDs in the combined file, and the counts corresponding to each block
def get_unique_blocks_counts():
    combined = np.load(COMBINED_FILE)
    uniques, counts = np.unique(combined, return_counts=True)
    return uniques, counts

# prints a dictionary with block names and the amount of that block\
def print_counts_blocknames(uniques, counts):
    names = []
    for id in uniques:
        names.append(str(id) + "=" + blockid_dict[id])
    print(dict(zip(np.array(names), counts)))

# compresses data based on compression_list. Every block id in a block "category" will be replaced with the first element of the category. For example, mossy cobblestone -> stone. diamond block -> iron block
def compress_data(data_list):
    # data_np_array = np.asarray(data_list)

    #compress
    for array in data_list:
        for list in compression_list:
            array[np.isin(array, list)] = list[0]

    return data_list

# scales compressed data to be from range 0 to num_categories, for one hot encoding
# this will remove our ability to visualize the compression unless we destransform
# i'm going to be lazy and just make this correspond to the index of the compression sublist (so air is 0, stone is 1, etc) since they're already in numerical order of starting element
def compressed_to_categorical(data_list):
    uniques = np.unique(np.asarray(data_list))
    print(len(uniques))
    uniques = np.sort(uniques)
    # map to 0-14, rather than random ids
    print(np.unique(data_list[0]))
    for array in data_list:
        for i, val in enumerate(uniques):
            array[array == val] = i
    print(np.unique(data_list[0]))
    return data_list

def round_generated_npy(dir):
    print("rounding")
    count = 1
    for dirname, _, filelist in os.walk(dir):
        for file in filelist:
            if file.startswith('generated') and file.endswith('.npy'):
                print(file)
                data = np.load(dirname + "/" + file)
                data[np.where(data >= .8)] = 1
                data[np.where(data < .8)] = 0
                np.save(dir + '/rounded/rounded_' + file, data)






# #  ========  preprocess ============
# data = load_data()
# dim_df = get_build_dim_df(data)
# filtered_data = cut_to_dim(dim_df, data, 32, 32, 32)
# padded_filtered_data = pad_arrays(filtered_data, 32, 32, 32)
# #
# # orig = np.copy(padded_filtered_data[100])
# compressed = compress_data(padded_filtered_data)
# compressed = compressed_to_categorical(compressed)
# print(compressed[0].shape)
# # sanity_check(orig, [compressed[100]])
# # round_generated_npy("generated_samples")
# # convert_to_schem("generated_samples/rounded/")
#
# # stone_only = convert_to_stone_only(padded_filtered_data)
# rotated = rotation_augmentation(compressed)
# flipped = reflection_augmentation(rotated)
# print(flipped[0].shape)
# # compress_and_write_data(padded_filtered_data, 'compressed_combined_rotated_flipped.npy')
# write_to_npy_file(flipped, "compressedcategorical_combined_rotated_flipped.npy")


# uniques, counts = get_unique_blocks_counts()
# print_counts_blocknames(uniques, counts)



# max_x, max_y, max_z = calc_max_axis(data)
# print("max x: ", max_x, "    max y: ", max_y, "     max z: ", max_z)
# scaled = scale_data(stone_only)
# sanity_check(stone_only[0], [scaled[0]])
# convert_to_schem(SANITY_CHECK_DIR)
# print(rotated[0].shape)
# test = reflection_augmentation([rotated[0]])
# print(len(test))
# sanity_check(test[0], test[1:])
# convert_to_schem(SANITY_CHECK_DIR)
# sanity_check(filtered_data[25], padded_filtered_data[25])
# dim_df2 = get_build_dim_df(filtered_data)
# dim_df2.hist(bins = 30)
# # dim_df.hist(bins=200)
# plt.show()
