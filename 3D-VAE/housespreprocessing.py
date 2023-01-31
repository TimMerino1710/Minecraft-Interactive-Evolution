import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from nbtschematic import SchematicFile
import importlib
import tensorflow as tf
from scipy.ndimage import zoom
from visualizer import binary_visualizer
from pprint import pprint
from json import JSONEncoder

VISUALIZER = binary_visualizer(16, 1, 4)

# blocckid_dict = {
# 0:["Air"],
# 1:["Stone","Granite","Polished Granite","Diorite","Polished Diorite","Andesite","Polished Andesite"],
# 2:["Grass"],
# 3:["Dirt","Coarse Dirt","Podzol"],
# 4:["Cobblestone"],
# 5:["Oak Wood Plank","Spruce Wood Plank","Birch Wood Plank","Jungle Wood Plank","Acacia Wood Plank","Dark Oak Wood Plank"],
# 6:["Oak Sapling","Spruce Sapling","Birch Sapling","Jungle Sapling","Acacia Sapling","Dark Oak Sapling"],
# 7:["Bedrock"],
# 8:["Flowing Water"],
# 9:["Still Water"],
# 10:["Flowing Lava"],
# 11:["Still Lava"],
# 12:["Sand","Red Sand"],
# 13:["Gravel"],
# 14:["Gold Ore"],
# 15:["Iron Ore"],
# 16:["Coal Ore"],
# 17:["Oak Wood","Spruce Wood","Birch Wood","Jungle Wood"],
# 18:["Oak Leaves","Spruce Leaves","Birch Leaves","Jungle Leaves"],
# 19:["Sponge","Wet Sponge"],
# 20:["Glass"],
# 21:["Lapis Lazuli Ore"],
# 22:["Lapis Lazuli Block"],
# 23:["Dispenser"],
# 24:["Sandstone","Chiseled Sandstone","Smooth Sandstone"],
# 25:["Note Block"],
# 26:["Bed"],
# 27:["Powered Rail"],
# 28:["Detector Rail"],
# 29:["Sticky Piston"],
# 30:["Cobweb"],
# 31:["Dead Shrub","Grass","Fern"],
# 32:["Dead Bush"],
# 33:["Piston"],
# 34:["Piston Head"],
# 35:["White Wool","Orange Wool","Magenta Wool","Light Blue Wool","Yellow Wool","Lime Wool","Pink Wool","Gray Wool","Light Gray Wool","Cyan Wool","Purple Wool","Blue Wool","Brown Wool","Green Wool","Red Wool","Black Wool"],
# 37:["Dandelion"],
# 38:["Poppy","Blue Orchid","Allium","Azure Bluet","Red Tulip","Orange Tulip","White Tulip","Pink Tulip","Oxeye Daisy"],
# 39:["Brown Mushroom"],
# 40:["Red Mushroom"],
# 41:["Gold Block"],
# 42:["Iron Block"],
# 43:["Double Stone Slab","Double Sandstone Slab","Double Wooden Slab","Double Cobblestone Slab","Double Brick Slab","Double Stone Brick Slab","Double Nether Brick Slab","Double Quartz Slab"],
# 44:["Stone Slab","Sandstone Slab","Wooden Slab","Cobblestone Slab","Brick Slab","Stone Brick Slab","Nether Brick Slab","Quartz Slab"],
# 45:["Bricks"],
# 46:["TNT"],
# 47:["Bookshelf"],
# 48:["Moss Stone"],
# 49:["Obsidian"],
# 50:["Torch"],
# 51:["Fire"],
# 52:["Monster Spawner"],
# 53:["Oak Wood Stairs"],
# 54:["Chest"],
# 55:["Redstone Wire"],
# 56:["Diamond Ore"],
# 57:["Diamond Block"],
# 58:["Crafting Table"],
# 59:["Wheat Crops"],
# 60:["Farmland"],
# 61:["Furnace"],
# 62:["Burning Furnace"],
# 63:["Standing Sign Block"],
# 64:["Oak Door Block"],
# 65:["Ladder"],
# 66:["Rail"],
# 67:["Cobblestone Stairs"],
# 68:["Wall"-"mounted Sign Block"],
# 69:["Lever"],
# 70:["Stone Pressure Plate"],
# 71:["Iron Door Block"],
# 72:["Wooden Pressure Plate"],
# 73:["Redstone Ore"],
# 74:["Glowing Redstone Ore"],
# 75:["Redstone Torch (off))"],
# 76:["Redstone Torch (on)"],
# 77:["Stone Button"],
# 78:["Snow"],
# 79:["Ice"],
# 80:["Snow Block"],
# 81:["Cactus"],
# 82:["Clay"],
# 83:["Sugar Canes"],
# 84:["Jukebox"],
# 85:["Oak Fence"],
# 86:["Pumpkin"],
# 87:["Netherrack"],
# 88:["Soul Sand"],
# 89:["Glowstone"],
# 90:["Nether Portal"],
# 91:["Jack o'Lantern"],
# 92:["Cake Block"],
# 93:["Redstone Repeater Block (off)"],
# 94:["Redstone Repeater Block (on)"],
# 95:["White Stained Glass","Orange Stained Glass","Magenta Stained Glass","Light Blue Stained Glass","Yellow Stained Glass","Lime Stained Glass","Pink Stained Glass","Gray Stained Glass","Light Gray Stained Glass","Cyan Stained Glass","Purple Stained Glass","Blue Stained Glass","Brown Stained Glass","Green Stained Glass","Red Stained Glass","Black Stained Glass"],
# 96:["Wooden Trapdoor"],
# 97:["Stone Monster Egg","Cobblestone Monster Egg","Stone Brick Monster Egg","Mossy Stone Brick Monster Egg","Cracked Stone Brick Monster Egg","Chiseled Stone Brick Monster Egg"],
# 98:["Stone Bricks","Mossy Stone Bricks","Cracked Stone Bricks","Chiseled Stone Bricks"],
# 99:["Brown Mushroom Block"],
# 100:["Red Mushroom Block"],
# 101:["Iron Bars"],
# 102:["Glass Pane"],
# 103:["Melon Block"],
# 104:["Pumpkin Stem"],
# 105:["Melon Stem"],
# 106:["Vines"],
# 107:["Oak Fence Gate"],
# 108:["Brick Stairs"],
# 109:["Stone Brick Stairs"],
# 110:["Mycelium"],
# 111:["Lily Pad"],
# 112:["Nether Brick"],
# 113:["Nether Brick Fence"],
# 114:["Nether Brick Stairs"],
# 115:["Nether Wart"],
# 116:["Enchantment Table"],
# 117:["Brewing Stand"],
# 118:["Cauldron"],
# 119:["End Portal"],
# 120:["End Portal Frame"],
# 121:["End Stone"],
# 122:["Dragon Egg"],
# 123:["Redstone Lamp (inactive)"],
# 124:["Redstone Lamp (active)"],
# 125:["Double Oak Wood Slab","Double Spruce Wood Slab","Double Birch Wood Slab","Double Jungle Wood Slab","Double Acacia Wood Slab","Double Dark Oak Wood Slab"],
# 126:["Oak Wood Slab","Spruce Wood Slab","Birch Wood Slab","Jungle Wood Slab","Acacia Wood Slab","Dark Oak Wood Slab"],
# 127:["Cocoa"],
# 128:["Sandstone Stairs"],
# 129:["Emerald Ore"],
# 130:["Ender Chest"],
# 131:["Tripwire Hook"],
# 132:["Tripwire"],
# 133:["Emerald Block"],
# 134:["Spruce Wood Stairs"],
# 135:["Birch Wood Stairs"],
# 136:["Jungle Wood Stairs"],
# 137:["Command Block"],
# 138:["Beacon"],
# 139:["Cobblestone Wall","Mossy Cobblestone Wall"],
# 140:["Flower Pot"],
# 141:["Carrots"],
# 142:["Potatoes"],
# 143:["Wooden Button"],
# 144:["Mob Head"],
# 145:["Anvil"],
# 146:["Trapped Chest"],
# 147:["Weighted Pressure Plate (light)"],
# 148:["Weighted Pressure Plate (heavy)"],
# 149:["Redstone Comparator (inactive)"],
# 150:["Redstone Comparator (active)"],
# 151:["Daylight Sensor"],
# 152:["Redstone Block"],
# 153:["Nether Quartz Ore"],
# 154:["Hopper"],
# 155:["Quartz Block","Chiseled Quartz Block","Pillar Quartz Block"],
# 156:["Quartz Stairs"],
# 157:["Activator Rail"],
# 158:["Dropper"],
# 159:["White Hardened Clay","Orange Hardened Clay","Magenta Hardened Clay","Light Blue Hardened Clay","Yellow Hardened Clay","Lime Hardened Clay","Pink Hardened Clay","Gray Hardened Clay","Light Gray Hardened Clay","Cyan Hardened Clay","Purple Hardened Clay","Blue Hardened Clay","Brown Hardened Clay","Green Hardened Clay","Red Hardened Clay","Black Hardened Clay"],
# 160:["White Stained Glass Pane","Orange Stained Glass Pane","Magenta Stained Glass Pane","Light Blue Stained Glass Pane","Yellow Stained Glass Pane","Lime Stained Glass Pane","Pink Stained Glass Pane","Gray Stained Glass Pane","Light Gray Stained Glass Pane","Cyan Stained Glass Pane","Purple Stained Glass Pane","Blue Stained Glass Pane","Brown Stained Glass Pane","Green Stained Glass Pane","Red Stained Glass Pane","Black Stained Glass Pane"],
# 161:["Acacia Leaves","Dark Oak Leaves"],
# 162:["Acacia Wood","Dark Oak Wood"],
# 163:["Acacia Wood Stairs"],
# 164:["Dark Oak Wood Stairs"],
# 165:["Slime Block"],
# 166:["Barrier"],
# 167:["Iron Trapdoor"],
# 168:["Prismarine","Prismarine Bricks","Dark Prismarine"],
# 169:["Sea Lantern"],
# 170:["Hay Bale"],
# 171:["White Carpet","Orange Carpet","Magenta Carpet","Light Blue Carpet","Yellow Carpet","Lime Carpet","Pink Carpet","Gray Carpet","Light Gray Carpet","Cyan Carpet","Purple Carpet","Blue Carpet","Brown Carpet","Green Carpet","Red Carpet","Black Carpet"],
# 172:["Hardened Clay"],
# 173:["Block of Coal"],
# 174:["Packed Ice"],
# 175:["Sunflower","Lilac","Double Tallgrass","Large Fern","Rose Bush","Peony"],
# 176:["Free"-"standing Banner"],
# 177:["Wall"-"mounted Banner"],
# 178:["Inverted Daylight Sensor"],
# 179:["Red Sandstone","Chiseled Red Sandstone","Smooth Red Sandstone"],
# 180:["Red Sandstone Stairs"],
# 181:["Double Red Sandstone Slab"],
# 182:["Red Sandstone Slab"],
# 183:["Spruce Fence Gate"],
# 184:["Birch Fence Gate"],
# 185:["Jungle Fence Gate"],
# 186:["Dark Oak Fence Gate"],
# 187:["Acacia Fence Gate"],
# 188:["Spruce Fence"],
# 189:["Birch Fence"],
# 190:["Jungle Fence"],
# 191:["Dark Oak Fence"],
# 192:["Acacia Fence"],
# 193:["Spruce Door Block"],
# 194:["Birch Door Block"],
# 195:["Jungle Door Block"],
# 196:["Acacia Door Block"],
# 197:["Dark Oak Door Block"],
# 198:["End Rod"],
# 199:["Chorus Plant"],
# 200:["Chorus Flower"],
# 201:["Purpur Block"],
# 202:["Purpur Pillar"],
# 203:["Purpur Stairs"],
# 204:["Purpur Double Slab"],
# 205:["Purpur Slab"],
# 206:["End Stone Bricks"],
# 207:["Beetroot Block"],
# 208:["Grass Path"],
# 209:["End Gateway"],
# 210:["Repeating Command Block"],
# 211:["Chain Command Block"],
# 212:["Frosted Ice"],
# 213:["Magma Block"],
# 214:["Nether Wart Block"],
# 215:["Red Nether Brick"],
# 216:["Bone Block"],
# 217:["Structure Void"],
# 218:["Observer"],
# 219:["White Shulker Box"],
# 220:["Orange Shulker Box"],
# 221:["Magenta Shulker Box"],
# 222:["Light Blue Shulker Box"],
# 223:["Yellow Shulker Box"],
# 224:["Lime Shulker Box"],
# 225:["Pink Shulker Box"],
# 226:["Gray Shulker Box"],
# 227:["Light Gray Shulker Box"],
# 228:["Cyan Shulker Box"],
# 229:["Purple Shulker Box"],
# 230:["Blue Shulker Box"],
# 231:["Brown Shulker Box"],
# 232:["Green Shulker Box"],
# 233:["Red Shulker Box"],
# 234:["Black Shulker Box"],
# 235:["White Glazed Terracotta"],
# 236:["Orange Glazed Terracotta"],
# 237:["Magenta Glazed Terracotta"],
# 238:["Light Blue Glazed Terracotta"],
# 239:["Yellow Glazed Terracotta"],
# 240:["Lime Glazed Terracotta"],
# 241:["Pink Glazed Terracotta"],
# 242:["Gray Glazed Terracotta"],
# 243:["Light Gray Glazed Terracotta"],
# 244:["Cyan Glazed Terracotta"],
# 245:["Purple Glazed Terracotta"],
# 246:["Blue Glazed Terracotta"],
# 247:["Brown Glazed Terracotta"],
# 248:["Green Glazed Terracotta"],
# 249:["Red Glazed Terracotta"],
# 250:["Black Glazed Terracotta"],
# 251:["White Concrete","Orange Concrete","Magenta Concrete","Light Blue Concrete","Yellow Concrete","Lime Concrete","Pink Concrete","Gray Concrete","Light Gray Concrete","Cyan Concrete","Purple Concrete","Blue Concrete","Brown Concrete","Green Concrete","Red Concrete","Black Concrete"],
# 252:["White Concrete Powder","Orange Concrete Powder","Magenta Concrete Powder","Light Blue Concrete Powder","Yellow Concrete Powder","Lime Concrete Powder","Pink Concrete Powder","Gray Concrete Powder","Light Gray Concrete Powder","Cyan Concrete Powder","Purple Concrete Powder","Blue Concrete Powder","Brown Concrete Powder","Green Concrete Powder","Red Concrete Powder","Black Concrete Powder"],
# 255:["Structure Block"],
# 256:["Iron Shovel"],
# 257:["Iron Pickaxe"],
# 258:["Iron Axe"],
# 259:["Flint and Steel"],
# 260:["Apple"],
# 261:["Bow"],
# 262:["Arrow"],
# 263:["Coal","Charcoal"],
# 264:["Diamond"],
# 265:["Iron Ingot"],
# 266:["Gold Ingot"],
# 267:["Iron Sword"],
# 268:["Wooden Sword"],
# 269:["Wooden Shovel"],
# 270:["Wooden Pickaxe"],
# 271:["Wooden Axe"],
# 272:["Stone Sword"],
# 273:["Stone Shovel"],
# 274:["Stone Pickaxe"],
# 275:["Stone Axe"],
# 276:["Diamond Sword"],
# 277:["Diamond Shovel"],
# 278:["Diamond Pickaxe"],
# 279:["Diamond Axe"],
# 280:["Stick"],
# 281:["Bowl"],
# 282:["Mushroom Stew"],
# 283:["Golden Sword"],
# 284:["Golden Shovel"],
# 285:["Golden Pickaxe"],
# 286:["Golden Axe"],
# 287:["String"],
# 288:["Feather"],
# 289:["Gunpowder"],
# 290:["Wooden Hoe"],
# 291:["Stone Hoe"],
# 292:["Iron Hoe"],
# 293:["Diamond Hoe"],
# 294:["Golden Hoe"],
# 295:["Wheat Seeds"],
# 296:["Wheat"],
# 297:["Bread"],
# 298:["Leather Helmet"],
# 299:["Leather Tunic"],
# 300:["Leather Pants"],
# 301:["Leather Boots"],
# 302:["Chainmail Helmet"],
# 303:["Chainmail Chestplate"],
# 304:["Chainmail Leggings"],
# 305:["Chainmail Boots"],
# 306:["Iron Helmet"],
# 307:["Iron Chestplate"],
# 308:["Iron Leggings"],
# 309:["Iron Boots"],
# 310:["Diamond Helmet"],
# 311:["Diamond Chestplate"],
# 312:["Diamond Leggings"],
# 313:["Diamond Boots"],
# 314:["Golden Helmet"],
# 315:["Golden Chestplate"],
# 316:["Golden Leggings"],
# 317:["Golden Boots"],
# 318:["Flint"],
# 319:["Raw Porkchop"],
# 320:["Cooked Porkchop"],
# 321:["Painting"],
# 322:["Golden Apple","Enchanted Golden Apple"],
# 323:["Sign"],
# 324:["Oak Door"],
# 325:["Bucket"],
# 326:["Water Bucket"],
# 327:["Lava Bucket"],
# 328:["Minecart"],
# 329:["Saddle"],
# 330:["Iron Door"],
# 331:["Redstone"],
# 332:["Snowball"],
# 333:["Oak Boat"],
# 334:["Leather"],
# 335:["Milk Bucket"],
# 336:["Brick"],
# 337:["Clay"],
# 338:["Sugar Canes"],
# 339:["Paper"],
# 340:["Book"],
# 341:["Slimeball"],
# 342:["Minecart with Chest"],
# 343:["Minecart with Furnace"],
# 344:["Egg"],
# 345:["Compass"],
# 346:["Fishing Rod"],
# 347:["Clock"],
# 348:["Glowstone Dust"],
# 349:["Raw Fish","Raw Salmon","Clownfish","Pufferfish"],
# 350:["Cooked Fish","Cooked Salmon"],
# 351:["Ink Sack","Rose Red","Cactus Green","Coco Beans","Lapis Lazuli","Purple Dye","Cyan Dye","Light Gray Dye","Gray Dye","Pink Dye","Lime Dye","Dandelion Yellow","Light Blue Dye","Magenta Dye","Orange Dye","Bone Meal"],
# 352:["Bone"],
# 353:["Sugar"],
# 354:["Cake"],
# 355:["Bed"],
# 356:["Redstone Repeater"],
# 357:["Cookie"],
# 358:["Map"],
# 359:["Shears"],
# 360:["Melon"],
# 361:["Pumpkin Seeds"],
# 362:["Melon Seeds"],
# 363:["Raw Beef"],
# 364:["Steak"],
# 365:["Raw Chicken"],
# 366:["Cooked Chicken"],
# 367:["Rotten Flesh"],
# 368:["Ender Pearl"],
# 369:["Blaze Rod"],
# 370:["Ghast Tear"],
# 371:["Gold Nugget"],
# 372:["Nether Wart"],
# 373:["Potion"],
# 374:["Glass Bottle"],
# 375:["Spider Eye"],
# 376:["Fermented Spider Eye"],
# 377:["Blaze Powder"],
# 378:["Magma Cream"],
# 379:["Brewing Stand"],
# 380:["Cauldron"],
# 381:["Eye of Ender"],
# 382:["Glistering Melon"],
# 383:["Spawn Elder Guardian","Spawn Wither Skeleton","Spawn Stray","Spawn Husk","Spawn Zombie Villager","Spawn Skeleton Horse","Spawn Zombie Horse","Spawn Donkey","Spawn Mule","Spawn Evoker","Spawn Vex","Spawn Vindicator","Spawn Creeper","Spawn Skeleton","Spawn Spider","Spawn Zombie","Spawn Slime","Spawn Ghast","Spawn Zombie Pigman","Spawn Enderman","Spawn Cave Spider","Spawn Silverfish","Spawn Blaze","Spawn Magma Cube","Spawn Bat","Spawn Witch","Spawn Endermite","Spawn Guardian","Spawn Shulker","Spawn Pig","Spawn Sheep","Spawn Cow","Spawn Chicken","Spawn Squid","Spawn Wolf","Spawn Mooshroom","Spawn Ocelot","Spawn Horse","Spawn Rabbit","Spawn Polar Bear","Spawn Llama","Spawn Parrot","Spawn Villager"],
# 384:["Bottle o' Enchanting"],
# 385:["Fire Charge"],
# 386:["Book and Quill"],
# 387:["Written Book"],
# 388:["Emerald"],
# 389:["Item Frame"],
# 390:["Flower Pot"],
# 391:["Carrot"],
# 392:["Potato"],
# 393:["Baked Potato"],
# 394:["Poisonous Potato"],
# 395:["Empty Map"],
# 396:["Golden Carrot"],
# 397:["Mob Head (Skeleton)","Mob Head (Wither Skeleton)","Mob Head (Zombie)","Mob Head (Human)","Mob Head (Creeper)","Mob Head (Dragon)"],
# 398:["Carrot on a Stick"],
# 399:["Nether Star"],
# 400:["Pumpkin Pie"],
# 401:["Firework Rocket"],
# 402:["Firework Star"],
# 403:["Enchanted Book"],
# 404:["Redstone Comparator"],
# 405:["Nether Brick"],
# 406:["Nether Quartz"],
# 407:["Minecart with TNT"],
# 408:["Minecart with Hopper"],
# 409:["Prismarine Shard"],
# 410:["Prismarine Crystals"],
# 411:["Raw Rabbit"],
# 412:["Cooked Rabbit"],
# 413:["Rabbit Stew"],
# 414:["Rabbit's Foot"],
# 415:["Rabbit Hide"],
# 416:["Armor Stand"],
# 417:["Iron Horse Armor"],
# 418:["Golden Horse Armor"],
# 419:["Diamond Horse Armor"],
# 420:["Lead"],
# 421:["Name Tag"],
# 422:["Minecart with Command Block"],
# 423:["Raw Mutton"],
# 424:["Cooked Mutton"],
# 425:["Banner"],
# 426:["End Crystal"],
# 427:["Spruce Door"],
# 428:["Birch Door"],
# 429:["Jungle Door"],
# 430:["Acacia Door"],
# 431:["Dark Oak Door"],
# 432:["Chorus Fruit"],
# 433:["Popped Chorus Fruit"],
# 434:["Beetroot"],
# 435:["Beetroot Seeds"],
# 436:["Beetroot Soup"],
# 437:["Dragon's Breath"],
# 438:["Splash Potion"],
# 439:["Spectral Arrow"],
# 440:["Tipped Arrow"],
# 441:["Lingering Potion"],
# 442:["Shield"],
# 443:["Elytra"],
# 444:["Spruce Boat"],
# 445:["Birch Boat"],
# 446:["Jungle Boat"],
# 447:["Acacia Boat"],
# 448:["Dark Oak Boat"],
# 449:["Totem of Undying"],
# 450:["Shulker Shell"],
# 452:["Iron Nugget"],
# 453:["Knowledge Book"],
# 2256:["13 Disc"],
# 2257:["Cat Disc"],
# 2258:["Blocks Disc"],
# 2259:["Chirp Disc"],
# 2260:["Far Disc"],
# 2261:["Mall Disc"],
# 2262:["Mellohi Disc"],
# 2263:["Stal Disc"],
# 2264:["Strad Disc"],
# 2265:["Ward Disc"],
# 2266:["11 Disc"],
# 2267:["Wait Disc"],
# }

from json import JSONEncoder
import json
# Takes a directory of .schem files, and converts them into a combined array of size (# samples, x, y, z), where each entry is a minecraft block name (ex: minecraft:air)
import random

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

mapping_df = pd.read_csv('../compression_csv7.csv')
id_to_type = {0:'1', 1:'2', 2:'3', 5:'PLANKS', 8:'WATER', 12:'SAND', 18:'LEAVES', 20:'GLASS', 22:'LAPIS_BLOCK', 37:'YELLOW_FLOWER', 43:'STONE_SLAB', 53:'OAK_STAIRS', 85:'FENCE'}
def create_combined_blockname_data(schematic_dir):
    data_list = []
    for file in os.listdir(schematic_dir):
        if file.endswith('.schem'):
            file_path = schematic_dir + '/' + file

            # load schem file
            schem = SchematicFile.load(file_path)

            # get block data, where each value corresponds to an index in the palette dictionary
            blockdata = schem.blocks.unpack()
            blockdata = blockdata.astype(object)

            # get the palette dictionary
            palette = schem.palette

            # reverse it so that the keys are indices and the values are the block names
            reverse_palette_dict = {y: x for x, y in palette.items()}

            # replace indices with their block names
            for key, value in reverse_palette_dict.items():

                # remove unwanted data from the palette string
                block_tag = reverse_palette_dict[key]
                block_tag = block_tag.partition(":")[2]
                if "[" in block_tag:
                    block_tag = block_tag.partition('[')[0]

                # print(mapping_df[mapping_df['block tag'] == block_tag])
                blockdata[blockdata == key] = block_tag

            data_list.append(blockdata)

    # print(len(data_list))
    combined = np.asarray(data_list)
    # print(combined.shape)
    uniques, counts = np.unique(combined, return_counts=True)
    # print(uniques)
    # print(counts)

    return combined
    # np.save(schematic_dir + "/combined_blocknames.npy", combined)

def compressed_to_categorical(data):
    # Get unique compressed block ids
    uniques = np.unique(data)
    print("Number of block categories: ", len(uniques))
    print("uniques: ", uniques)

    # sort them so they're lowest to highest
    uniques = np.sort(uniques)
    print("sorted unqiues: ", uniques)
    # map to 1-10, rather than random ids
    for i, category in enumerate(uniques):
        print(i, ", category: ", category)
        data[data == category] = i
    return data


def convert_to_blockid(combined_array):
    uniques = np.unique(combined_array)

    # for each unique value in our combined array, replace the block name with numerical id
    for val in uniques:

        # get block id for this value
        block_id = mapping_df[mapping_df['block tag'] == val]['compressed blockid']
        if len(block_id) > 0:
            # print(val)
            block_id = int(block_id.iloc[0])
        else:
            block_id = "NOT FOUND"
            print(val, " ", block_id)

        # replace all instances of this value with the corresponding block id
        combined_array[np.where(combined_array == val)] = block_id

    return combined_array

# compress to a smaller set of blocks (without having to edit the csv)
def smaller_compression(data):
    uniques = np.unique(data)

    # create mapping to combine some categories
    # stairs turn into slabs, leaves turn into air (deleted), glass turns into air (deleted), metal turns into stone, sand turns into dirt, fences into wood
    smaller_compression_mapping = {53:126, 18:0, 20:0, 42:1, 12:2, 85:5}

    for key in smaller_compression_mapping:
        data[data == key] = smaller_compression_mapping[key]

    return data


# takes a compressed np array, then prints the block name and its count in the data
def print_blockname_counts_compressed(data):
    uniques, counts = np.unique(data, return_counts=True)
    blockname_counts_dict = {}
    for val, count in zip(uniques, counts):
        # get the block name from the categories subtable
        row = mapping_df.loc[mapping_df['category id'] == val]
        block_name = row['category block name'].iloc[0]

        blockname_counts_dict[block_name] = count
    pprint(blockname_counts_dict)

    # print sorted
    # print(sorted(blockname_counts_dict.items(), key=lambda x:x[1]))


def compress_craftassist(data_np):
    uniques = np.unique(data_np)
    # for each unique value in our combined array, replace the block id with the compressed block id
    for val in uniques:
        # get block id for this value
        block_id = mapping_df[mapping_df['block id'] == val]['compressed blockid']
        if len(block_id) > 0:
            # print(val)
            block_id = int(block_id.iloc[0])
        else:
            block_id = "NOT FOUND"
            print(val, " ", block_id)

        # replace all instances of this value with the corresponding block id
        data_np[np.where(data_np == val)] = block_id
    return data_np


def cropHouse(h):
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(h)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)
    out = h[top_left[0]:bottom_right[0] + 1,  # plus 1 because slice isn't
          top_left[1]:bottom_right[1] + 1, top_left[2]:bottom_right[2] + 1]  # inclusive

    return out

def houseTrans(h, s=(16, 16, 16)):
    hts = []

    s2 = cropHouse(h)
    # print("cropped house shape: ", s2.shape)
    s2s = s2.shape
    ds = (s[0] - s2s[0], s[1] - s2s[1], s[2] - s2s[2])
    # print(ds)
    # for x in range(1):
    for x in range(0, s[0] - s2s[0] + 1, 2):
        # for y in range(1):
        for y in range(0, s[1] - s2s[1], 2):
            for z in range(1):
                # for z in range(s[2]-s2s[2]+1):
                thouse = np.zeros(shape=s)
                thouse[x:x + s2s[0], y:y + s2s[1], z:z + s2s[2]] = s2.copy()
                hts.append(thouse)
    # print("len of hts: ", len(hts))
    return hts

def transpose_houses(house, out_shape=(16, 16, 16), steps=1):
    transposes_houses = []

    cropped_house = cropHouse(house)
    # print("cropped house shape: ", s2.shape)
    cropped_shape = cropped_house.shape

    # transpose along x axis, skipping by 2
    for x in range(0, out_shape[0] - cropped_shape[0] + 1, steps):
        # transpose along y axis, skipping by 2
        for y in range(0, out_shape[1] - cropped_shape[1], steps):
            # create empty array of desired output shape
            transposed_house = np.zeros(shape=out_shape)

            # copy the cropped house to this chunk of the array
            transposed_house[x:x + cropped_shape[0], y:y + cropped_shape[1], 0:cropped_shape[2]] = cropped_house.copy()
            transposes_houses.append(transposed_house)
    # print("len of hts: ", len(hts))
    return transposes_houses


def transpose_and_stretch_house(house, out_shape=(16, 16, 16), steps=1):
    transposes_and_stretched_houses = []

    # for house in data:
    cropped_house = cropHouse(house)
    cropped_shape = cropped_house.shape
    # print("cropped shape: ", cropped_shape)
    # get transpositions of normal house
    transposes_and_stretched_houses += transpose_houses(house, out_shape, steps)
    transpose_only_ct = len(transposes_and_stretched_houses)

    # if we can fit after stretching in any direction, get the stretched house and all possible transpositions of it
    if cropped_shape[0] * 2 <= out_shape[0]:
        stretched_x_house = cropped_house.repeat(2, axis=0)
        stretched_x_houses = transpose_houses(stretched_x_house, out_shape, steps)
        # print("number of stretched x houses", len(stretched_x_houses))
        transposes_and_stretched_houses += stretched_x_houses
    if cropped_shape[1] * 2 <= out_shape[1]:
        stretched_y_house = cropped_house.repeat(2, axis=1)
        stretched_y_houses = transpose_houses(stretched_y_house, out_shape, steps)
        # print("number of stretched y houses", len(stretched_y_houses))
        transposes_and_stretched_houses += stretched_y_houses

    # stretching on axis 2 stretches up, creating towers. This leads to less diversity, it seems.
    # if cropped_shape[2] * 2 <= out_shape[2]:
    #     stretched_z_house = cropped_house.repeat(2, axis=2)
    #     stretched_z_houses = transpose_houses(stretched_z_house, out_shape)
    #     print("number of stretched z houses", len(stretched_z_houses))
    #     transposes_and_stretched_houses += stretched_z_houses


    # print("number from stretch augment only: ", len(transposes_and_stretched_houses) - transpose_only_ct)
    # print("number of stretched transposed houses: ", len(transposes_and_stretched_houses))
    return transposes_and_stretched_houses


def augment_ingame(combined):
    HOUSE_DATASET = []

    TRANS_HOUSES = []

    for h in combined:
        # houses look rotated... just rotate them back
        h = np.rot90(h, axes=(0, 2))

        # remove bottom layer (got the ground as well) - i can't believe i got it right on the first try...
        h = h[3:, 3:, 1:-2]
        HOUSE_DATASET.append(h)

        # transpose and stretch
        tds = transpose_and_stretch_house(h, (16, 16, 16), steps=3)

        # transpose only:
        # tds = transpose_houses(h, (16, 16, 16), steps=2)
        # rotated
        for haus in tds:
            TRANS_HOUSES.append(haus)

    TRANS_HOUSES = rotation_augmentation(TRANS_HOUSES)
    TRANS_HOUSES = np.array(TRANS_HOUSES)

    print("\n \n Length of full augmented (ingame) houses: ", len(TRANS_HOUSES), "\n \n")
    return TRANS_HOUSES


def rotation_augmentation(data):
    rotated_data = []
    for array in data:
        rotated_data.append(np.rot90(array, axes=(1, 2)))
        # rotated_180 = np.rot90(rotated_90, axes=(1, 2))
        # rotated_data.append(np.rot90(array, 3, axes=(1, 2)))
        rotated_data.append(array)
        # rotated_data.append(rotated_90)
        # rotated_data.append(rotated_180)
        # rotated_data.append(rotated_270)
    return rotated_data


def augment_craftassist(data):
    augmented_houses = []
    for house in data:
        # rotate the craftassist data to proper orientation (based on renderings from massrender)
        house = np.rot90(house, 1, axes=(0, 2))

        # add the base house to the augmented set we return
        augmented_houses.append(house)

        # perform transpose only augmentation
        # transposed_houses = transpose_houses(house, (16, 16, 16), steps=2)

        # transpose and stretch
        transposed_houses = transpose_and_stretch_house(house, (16, 16, 16), steps=3)
        for th in transposed_houses:
            augmented_houses.append(th)

    augmented_houses = rotation_augmentation(augmented_houses)
    augmented_houses = np.array(augmented_houses)
    print("\n \n Length of full augmented (craftassist) houses: ", len(augmented_houses), "\n \n")
    return augmented_houses


def scale(data, factor):
    zoomed_data = []
    for x in data:
        zoomed = x.repeat(factor, axis=0).repeat(factor, axis=1).repeat(factor, axis=2)
        zoomed_data.append(zoomed)

    return np.array(zoomed_data)


# takes a percentage which will be the number of voxels flipped, adds one copy of noisy data for each data point
def noise_augment(data, percent):
    voxels = data[0].shape[0] * data[0].shape[1] * data[0].shape[2]
    on = voxels - int(voxels * percent)
    off = voxels - on
    print("num on voxels: ", on)
    print("num off voxels: ", off)
    noise_vector = np.array([0] * on + [1] * off)
    noise_vector.astype('bool')
    for i, struct in enumerate(data):
        print(i, "/", len(data))
        np.random.shuffle(noise_vector)
        noise_vector = noise_vector.reshape(struct.shape)
        noisy_struct = np.logical_not(struct, noise_vector)
        np.append(data, noisy_struct)

    return data


# gives the dimensions of the cropped house
def get_house_min_dims(house):
    true_points = np.argwhere(house)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    out = house[top_left[0]:bottom_right[0] + 1,  # plus 1 because slice isn't
          top_left[1]:bottom_right[1] + 1, top_left[2]:bottom_right[2] + 1]  # inclusive

    print(top_left)
    print(bottom_right)


# just stretching in one direction gives very tiny structures. Need to scale up and then stretch?
# def stretch_augment(data, factor, out_shape):
#     stretched_houses = []
#     for house in data:
#         original_shape = data.shape
#         cropped_house = cropHouse(house)
#         # if we can stretch in the x direction without going out of bounds:
#         # if cropped_house.shape[0] * factor <= out_shape[0]:
#
#         # stretch in x
#         stretched_x = house.repeat(factor, axis=0)
#         stretched_x = np.pad(stretched_x, ((0, 0), (24, 24), (24, 24)), mode='constant')
#         # stretch in y
#         stretched_y = house.repeat(factor, axis=1)
#         stretched_y = np.pad(stretched_y, ((24, 24), (0, 0), (24, 24)), mode='constant')
#         # stretch in z
#         stretched_z = house.repeat(factor, axis=2)
#         stretched_z = np.pad(stretched_z, ((24, 24), (24, 24), (0, 0)), mode='constant')
#
#         stretched_houses = stretched_houses + [stretched_x, stretched_y, stretched_z]
#     return np.asarray(stretched_houses)



# bads = [44, 50, 72, 74, 79, 81, 98, 110, 115, 117, 154, 157, 175, 177, 190, 192, 194, 200, 202, 207, 218, 228, 232, 237, 238, 250, 268, 284, 303, 304, 305, 320, 322, 334, 335, 340, 344, 357, 372, 391, 406, 422, 450, 478, 486, 491, 522, 539, 552, 553, 564, 568, 570, 574, 590, 594, 597, 605, 607, 609, 613, 616, 618, 625, 635, 650, 655, 656, 659, 660, 677, 683, 696, 705, 733, 739, 741, 743, 744, 751, 766, 783, 798, 809, 840, 841, 850, 854, 881, 883, 891, 902, 906, 908, 909, 913, 919, 921, 924, 929, 937, 940, 942, 951, 958, 979, 985, 995, 999, 1003, 1006, 1017, 1019, 1026]
# maybes = [19, 20, 21, 38, 42, 49, 51, 61, 65, 67, 86, 95, 116, 119, 126, 136, 140, 150, 159, 171, 178, 186, 188, 204, 209, 236, 249, 266, 272, 277, 280, 282, 289, 302, 339, 352, 354, 358, 359, 361, 362, 363, 367, 376, 378, 379, 393, 400, 408, 412, 427, 437, 451, 468, 477, 480, 481, 490, 499, 508, 541, 544, 549, 550, 556, 560, 565, 567, 586, 603, 604, 611, 644, 651, 653, 657, 658, 662, 663, 679, 684, 684, 686, 691, 701, 707, 708, 718, 725, 732, 740, 761,  771, 793, 824, 845, 847, 857, 863, 866, 876, 878, 886, 915, 925, 930, 945, 949, 956, 966, 970, 972, 975, 978, 984, 988, 993, 998, 1004, 1010, 1011, 1030, 1033, 1034, 1035, 1038]
good_idxs = [1, 100, 1005, 101, 1021, 1030, 1032, 1036, 111, 119, 126, 14, 154, 150, 178, 194, 193, 205, 203, 224, 212, 244, 228, 259, 258, 257, 246, 275, 274, 271, 27, 267, 266, 264, 287, 282, 307, 303, 300, 3, 296, 325, 322, 320, 316, 314, 310, 31, 309, 340, 339, 332, 355, 35, 348, 347, 346, 343, 370, 363, 361, 358, 393, 391, 383, 376, 400, 398, 425, 421, 419, 413, 412, 438, 433, 43, 427, 426, 453, 451, 450, 447, 470, 466, 462, 461, 458, 456, 481, 480, 50, 5, 495, 490, 489, 488, 487, 514, 513, 502, 528, 522, 541, 54, 534, 552, 57, 569, 578, 579, 58, 588, 590, 593, 596, 598, 60, 603, 604, 609, 612, 624, 629, 63, 630, 637, 638, 642, 644, 646, 662, 663, 666, 667, 669, 671, 677, 683, 687, 688, 694, 701, 703, 704, 705, 707, 709, 712, 715, 717, 721, 724, 736, 737, 739, 740, 742, 743, 744, 746, 748, 751, 752, 763, 764, 767, 771, 772, 776, 783, 788, 789, 79, 793, 795, 799, 8, 801, 807, 809, 810, 812, 818, 821, 830, 838, 841, 845, 85, 855, 860, 863, 871, 872, 875, 879, 883, 885, 899, 905, 906, 910, 912, 913, 916, 925, 929, 93, 930, 934, 942, 949, 956, 957, 960, 972, 977, 980, 997]
remove_idxs = [1037, 108, 128, 13, 134, 136, 151, 162, 167, 169, 201, 204, 220, 222, 234, 236, 238, 243, 245, 25, 26, 269, 272, 277, 278, 289, 304, 319, 336, 337, 338, 351, 353, 364, 365, 37, 373, 385, 399, 415, 429, 443, 469, 494, 500, 505, 533, 549, 560, 561, 571, 575, 577, 580, 595, 599, 600, 608, 61, 611, 615, 618, 62, 626, 635, 649, 653, 654, 657, 658, 673, 679, 690, 699, 723, 729, 730, 732, 733, 74, 753, 769, 782, 792, 82, 820, 829, 832, 857, 859, 866, 876, 88, 881, 882, 886, 891, 893, 896, 90, 907, 91, 911, 92, 926, 945, 950, 96, 963, 967, 97, 98, 981, 988, 1014, 1015, 1016, 1031, 1035, 107, 109, 118, 121, 123, 140, 149, 168, 170, 177, 186, 19, 199, 206, 217, 223, 230, 232, 247, 251, 276, 288, 302, 308, 312, 315, 317, 323, 335, 369, 380, 382, 386, 387, 389, 39, 390, 394, 401, 403, 404, 417, 423, 430, 434, 448, 457, 47, 485, 493, 496, 497, 504, 512, 520, 550, 553, 558, 559, 564, 568, 572, 574, 591, 606, 607, 613, 643, 65, 651, 655, 656, 66, 660, 675, 68, 68, 681, 686, 695, 70, 700, 71, 716, 722, 73, 749, 758, 778, 805, 824, 826, 835, 840, 843, 852, 854, 861, 888, 897, 900, 914, 918, 924, 933, 937, 939, 941, 944, 95, 953, 958, 962, 968, 973, 974, 991, 994, 995, 996, 999]

#########################
# Craftassist data (binary):
#########################

# craftassist = np.load('H:/16x16x16_binary_outliersremoved.npy')
#
# # # remove houses that were deemed "bad"
# # good_craftassist = []
# # for i, house in enumerate(craftassist):
# #     if i not in remove_idxs:
# #         good_craftassist.append(house)
#
# good_craftassist = []
# # select only houses that were deemed "good"
# for i, house in enumerate(craftassist):
#     if i in good_idxs:
#         good_craftassist.append(house)
# craftassist = np.asarray(good_craftassist)
# augmented_ca = augment_craftassist(craftassist)
# zoomed_ca = scale(augmented_ca, 4)
# print("Number of augmented craftassist houses: ", len(zoomed_ca))


#########################
# Craftassist data (categorical):
#########################
craftassist = np.load('H:/craftassist_notcompressed_largestCC_1039.npy')

# # remove houses that were deemed "bad"
# good_craftassist = []
# for i, house in enumerate(craftassist):
#     if i not in remove_idxs and len(np.unique(house) > 1):
#         good_craftassist.append(house)

good_craftassist = []
# select only houses that were deemed "good"
for i, house in enumerate(craftassist):
    if i in good_idxs:
        good_craftassist.append(house)
# do processing as categorical, before one hot encoding

# compress
craftassist = np.asarray(good_craftassist)
craftassist = compress_craftassist(craftassist)
craftassist = smaller_compression(craftassist)


# for some reason I still have some that are like empty? idk
good_craftassist = []
for house in craftassist:
    if len(np.unique(house)) > 1:
        good_craftassist.append(house)
craftassist = np.asarray(good_craftassist)


# augment
craftassist = augment_craftassist(craftassist)
craftassist = scale(craftassist, 4)

#########################
# ingame data:
#########################
# load and compress
combined_ig = create_combined_blockname_data('../ingame house schematics')
combined_ig = convert_to_blockid(combined_ig)
combined_ig = smaller_compression(combined_ig)

# augment
combined_ig = augment_ingame(combined_ig)
combined_ig = scale(combined_ig, 4)

# print(u)
# print(c)
# categorical_ig = compressed_to_categorical(converted_ig)
# categorical_ig[categorical_ig != 0] = 1
# augmented_ig = augment_ingame(categorical_ig)
# zoomed_data_ig = scale(augmented_ig, 4)
# print("Number of augmented craftassist houses: ", len(zoomed_data_ig))
# # np.save("H:\ingame_stretched_xy.npy", zoomed_data_ig)


########################
# combine data (categorical):
########################
combined_ig = np.concatenate((craftassist, combined_ig))
print("joined before turning categorical")

combined_ig = compressed_to_categorical(combined_ig)
combined_ig = combined_ig.astype(int)
# np.save("H:\joined_cureated_categorical.npy", joined)
print("joined after turning categorical")

combined_ig = tf.one_hot(combined_ig, 5, dtype=tf.int8).numpy()
print(combined_ig.shape)

np.save("H:\joined_cureated_stretched_rotated_categorical_onehot.npy", combined_ig)

# joined_categorical = np.argmax(joined, axis=4)
# u, c = np.unique(joined, return_counts=True)
# print(u)
# print(c)


#########################
# combine data (binary):
#########################
# joined = np.concatenate((zoomed_ca, zoomed_data_ig))
# np.save("H:\joined_curatedCA.npy", joined)


# sanity testing stretch directions
# for ax in range(0, 3)
# i=2
# og = categorical_ig[6]
# og = np.rot90(og, axes=(0, 2))
# og = og[3:, 3:, 1:-2]
# og_stretched_ax0 = transpose_and_stretch_house(og, (16, 16, 16))
# samples = np.asarray([og] + og_stretched_ax0[:3])
#     # samples = np.asarray([og, stretched_x_house])
# VISUALIZER.draw(samples, 'sanitycheck_stretchaxis' + str(i) + '.png')




# compare a sample from craftassist and ingame to make sure they're oriented the same
# take one near the end of each



# print(get_house_min_dims(augmented[0]))
# stretched = stretch_augment(augmented, 4)


# print(get_house_min_dims(zoomed_data[0]))

# noise_augmented = noise_augment(zoomed_data, percent=.10)

# sanity check
# randoms = []
# for i in range(0, 9):
#     randoms.append(random.randint(0, len(zoomed_data)))
# samples = zoomed_data[randoms]
# #
# VISUALIZER.draw(samples, 'sanitycheck_stretchaugmentation3.png')
# onehot = tf.one_hot(augmented, 10, dtype=tf.int8).numpy()

# print(mapping_df)
