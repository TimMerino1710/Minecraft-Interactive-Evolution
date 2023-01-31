import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from nbtschematic import SchematicFile
from pprint import pprint
import json
import tensorflow as tf
import cc3d

HOUSE_NUMPY_DIR = '../house_numpy_files'
HOUSE_OUT_DIR = 'H:/'
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


# returns the unique block IDs in the combined file, and the counts corresponding to each block
def get_unique_blocks_counts(file):
    # combined = np.load(COMBINED_FILE)
    combined = np.load(file)
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
    # print(np.unique(data_list[0]))
    for array in data_list:
        for i, val in enumerate(uniques):
            array[array == val] = i
    # print(np.unique(data_list[0]))
    return data_list


# performs compression but only deletes the air blocks
def compress_data_air_only(data_list):
    # data_np_array = np.asarray(data_list)

    #compress
    for array in data_list:
        air_list = compression_list[0]
        array[np.isin(array, air_list)] = air_list[0]

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

def cropHouse(h):
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(h)
    # take the smallest points and use them as the top left of your crop
    # print(h.shape)
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)
    out = h[top_left[0]:bottom_right[0] + 1,  # plus 1 because slice isn't
          top_left[1]:bottom_right[1] + 1, top_left[2]:bottom_right[2] + 1]  # inclusive

    return out

def transpose_houses(house, out_shape=(16, 16, 16)):
    transposes_houses = []

    cropped_house = cropHouse(house)
    # print("cropped house shape: ", s2.shape)
    cropped_shape = cropped_house.shape

    # transpose along x axis, skipping by 2
    for x in range(0, out_shape[0] - cropped_shape[0] + 1):
        # transpose along y axis, skipping by 2
        for y in range(0, out_shape[1] - cropped_shape[1]):
            # create empty array of desired output shape
            transposed_house = np.zeros(shape=out_shape)

            # copy the cropped house to this chunk of the array
            transposed_house[x:x + cropped_shape[0], y:y + cropped_shape[1], 0:cropped_shape[2]] = cropped_house.copy()
            transposes_houses.append(transposed_house)
    # print("len of hts: ", len(hts))
    return transposes_houses


def transpose_and_stretch_house(house, out_shape=(16, 16, 16)):
    transposes_and_stretched_houses = []

    # for house in data:
    cropped_house = cropHouse(house)
    cropped_shape = cropped_house.shape

    # get transpositions of normal house
    transposes_and_stretched_houses += transpose_houses(house, out_shape)

    # if we can fit after stretching in any direction, get the stretched house and all possible transpositions of it
    if cropped_shape[0] * 2 <= out_shape[0]:
        stretched_x_house = cropped_house.repeat(2, axis=0)
        stretched_x_houses = transpose_houses(stretched_x_house, out_shape)
        print("number of stretched x houses", len(stretched_x_houses))
        transposes_and_stretched_houses += stretched_x_houses
    if cropped_shape[1] * 2 <= out_shape[1]:
        stretched_y_house = cropped_house.repeat(2, axis=1)
        stretched_y_houses = transpose_houses(stretched_y_house, out_shape)
        print("number of stretched y houses", len(stretched_y_houses))
        transposes_and_stretched_houses += stretched_y_houses
    if cropped_shape[2] * 2 <= out_shape[2]:
        stretched_z_house = cropped_house.repeat(2, axis=2)
        stretched_z_houses = transpose_houses(stretched_z_house, out_shape)
        print("number of stretched z houses", len(stretched_z_houses))
        transposes_and_stretched_houses += stretched_z_houses



    print("number of stretched transposed houses: ", len(transposes_and_stretched_houses))
    return transposes_and_stretched_houses


def augment(combined):
    HOUSE_DATASET = []

    TRANS_HOUSES = []

    for h in combined:
        # houses look rotated... just rotate them back
        h = np.rot90(h, axes=(0, 2))

        # remove bottom layer (got the ground as well) - i can't believe i got it right on the first try...
        h = h[3:, 3:, 1:-2]
        HOUSE_DATASET.append(h)

        tds = transpose_and_stretch_house(h, (16, 16, 16))

        # rotated
        for haus in tds:
            TRANS_HOUSES.append(haus)



    TRANS_HOUSES = np.array(TRANS_HOUSES)

    print("\n \n Length of full augmented houses: ", len(TRANS_HOUSES), "\n \n")
    return TRANS_HOUSES


def zscore(dat,thresh=3):
    mean = np.mean(dat,axis=0)
    std =np.std(dat,axis=0)
    # print(std)

    outliers = []
    for x in dat:
        z_score = (x - mean)/std
        zs_abs = np.abs(np.linalg.norm(z_score))
        if zs_abs > thresh:
            outliers.append(x)
    return outliers


def remove_outliers(data):
    house_xyz = []
    for h in data:
        xyz = np.argwhere(h).squeeze()
        house_xyz.append(xyz)

    removed_outliers_data = []
    # print(np.unique(data[1]))
    for i, house in enumerate(house_xyz):
        # print(i)
        # if i not in [22, 872, 1760, 2164, 178]:
        outliers = zscore(house, 3)
        new_house = data[i]
        for o in outliers:
            new_house[o[0], o[1], o[2]] = 0

        removed_outliers_data.append(new_house)
        # else:
        #     print(np.unique(house))
    return removed_outliers_data


# removes all but the largest connected component from the data
def get_largest_connected_component(data):
    cc_houses = []

    for house in data:
        # create a copy of the house that is binarized, to simplify connected component labeling
        binarized = np.copy(house)
        binarized[binarized != 0] = 1

        # get the largest connected component in the binarized verison
        largest_cc = cc3d.largest_k(binarized, k=1, connectivity=26)

        # use the binarized connected component as a mask to get the largest connected component of the original data
        cc_masked_house = house * largest_cc

        cc_houses.append(cc_masked_house)

    return cc_houses






bads = [44, 50, 72, 74, 79, 81, 98, 110, 115, 117, 154, 157, 175, 177, 190, 192, 194, 200, 202, 207, 218, 228, 232, 237, 238, 250, 268, 284, 303, 304, 305, 320, 322, 334, 335, 340, 344, 357, 372, 391, 406, 422, 450, 478, 486, 491, 522, 539, 552, 553, 564, 568, 570, 574, 590, 594, 597, 605, 607, 609, 613, 616, 618, 625, 635, 650, 655, 656, 659, 660, 677, 683, 696, 705, 733, 739, 741, 743, 744, 751, 766, 783, 798, 809, 840, 841, 850, 854, 881, 883, 891, 902, 906, 908, 909, 913, 919, 921, 924, 929, 937, 940, 942, 951, 958, 979, 985, 995, 999, 1003, 1006, 1017, 1019, 1026]
maybes = [19, 20, 21, 38, 42, 49, 51, 61, 65, 67, 86, 95, 116, 119, 126, 136, 140, 150, 159, 171, 178, 186, 188, 204, 209, 236, 249, 266, 272, 277, 280, 282, 289, 302, 339, 352, 354, 358, 359, 361, 362, 363, 367, 376, 378, 379, 393, 400, 408, 412, 427, 437, 451, 468, 477, 480, 481, 490, 499, 508, 541, 544, 549, 550, 556, 560, 565, 567, 586, 603, 604, 611, 644, 651, 653, 657, 658, 662, 663, 679, 684, 684, 686, 691, 701, 707, 708, 718, 725, 732, 740, 761,  771, 793, 824, 845, 847, 857, 863, 866, 876, 878, 886, 915, 925, 930, 945, 949, 956, 966, 970, 972, 975, 978, 984, 988, 993, 998, 1004, 1010, 1011, 1030, 1033, 1034, 1035, 1038]
# #  ========  preprocess ============
# Load all of the craftassist npy files
data = load_data()

# # create a dataframe with the dimensions of each building
# dim_df = get_build_dim_df(data)
#
# # filter down to only those less than or equal to 16x16x16
# # This results in only 737 builds
# filtered_data = cut_to_dim(dim_df, data, 16, 16, 16)
# print(len(filtered_data))

# # compress data
# compressed = compress_data(data)

# we want a version that doesn't use old compression, but we can trust the air compression
# if we don't do at least the air compression, we end up with a different number of houses returned from remove outliers
compressed = compress_data_air_only(data)
#
# # get the largest connected component of each house
# connected_compressed = get_largest_connected_component(compressed)
#
# # get the cropped version of each house
# cropped_houses = []
# for house in connected_compressed:
#     print(house.shape)
#     cropped_houses.append(cropHouse(house))
#
# # filter down to houses that fit in 16x16x16 (and bigger than 5x5x5) after removing outliers
# filtered_houses = []
# for house in cropped_houses:
#     house_shape = house.shape
#     if house_shape[0] <= 16 and house_shape[1] <= 16 and house_shape[2] <= 16 and (house_shape[0] > 4 and house_shape[1] > 4 and house_shape[2] > 4):
#         filtered_houses.append(house)
#
# # pad all houses to be 16x16x16
# padded = pad_arrays(filtered_houses, 16, 16, 16)
#
# compressed_np = np.asarray(padded)
# u, c = np.unique(compressed_np, return_counts=True)
# print("uniques, counts for all data")
# print(u)
# print(c)

# compressed = compress_data(data)
# compressed = compressed_to_categorical(compressed)
# for i, house in enumerate(compressed):
#     house[house != 0] = 1
#     # some houses are either all 0 or all 1. We can remove them here
#     # for some reason pop and remove don't work
#     if len(np.unique(house)) == 2:
#         binarized.append(house)

# getting compressed versions but still removing the ones that cant be binarized so we end up with the same number, and so the curated list of indices still works
binarized = []
for i, house in enumerate(compressed):
    b = np.copy(house)
    b[b != 0] = 1
    # some houses are either all 0 or all 1. We can remove them here
    # for some reason pop and remove don't work
    if len(np.unique(b)) == 2:
        binarized.append(house)

# new filtering
outliers_removed_data = remove_outliers(binarized)

# crop each house to its minimum size after removing outliers
cropped_houses = []
for house in outliers_removed_data:
    cropped_houses.append(cropHouse(house))

# filter down to houses that fit in 16x16x16 (and bigger than 5x5x5) after removing outliers
filtered_houses = []
for house in cropped_houses:
    house_shape = house.shape
    if house_shape[0] <= 16 and house_shape[1] <= 16 and house_shape[2] <= 16 and (house_shape[0] > 4 and house_shape[1] > 4 and house_shape[2] > 4):
        filtered_houses.append(house)

print("Number of houses within 16x16x16 after removing outliers: ", len(filtered_houses))

# Pad the houses so they're all 16x16x16 (this may not be necessary)
padded_filtered_data = pad_arrays(filtered_houses, 16, 16, 16)
print(np.unique(np.asarray(padded_filtered_data)))
# compressed_categorical = compressed_to_categorical(padded_filtered_data)
largest_cc = get_largest_connected_component(padded_filtered_data)
print(np.unique(np.asarray(largest_cc)))
print(len(largest_cc))
write_to_npy_file(largest_cc, "craftassist_notcompressed_largestCC_1039.npy")



# #
# # orig = np.copy(padded_filtered_data[100])

# compressed = compress_data(padded_filtered_data)
# compressed = compressed_to_categorical(compressed)
# # # print(compressed[0].shape)
# # # sanity_check(orig, [compressed[100]])
# # # round_generated_npy("generated_samples")
# # # convert_to_schem("generated_samples/rounded/")
# #
# # binarize
# compressed = np.asarray(compressed)
# compressed[compressed != 0] = 1
# # binary = np.asarray(compressed)
# print(compressed.shape)
# write_to_npy_file(compressed, "16x16x16_craftassist.npy")
# stone_only = convert_to_stone_only(padded_filtered_data)
# rotated = rotation_augmentation(compressed)
# flipped = reflection_augmentation(compressed)
#
# one_hot = tf.one_hot(flipped, 15, dtype=tf.int8).numpy()
# # print(len(one_hot))
# # print(one_hot[0].shape)
# # compress_and_write_data(padded_filtered_data, 'compressed_combined_rotated_flipped.npy')
# print(len(one_hot))
# write_to_npy_file(one_hot, "16_onehot_flipped.npy")


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
