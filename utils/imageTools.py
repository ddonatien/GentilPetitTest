import torch
import numpy as np
from PIL import Image

# TODO sort out input outpout format so that all functions accept same objects

def to_tiles(image, tile_size):
    if type(image) == np.ndarray or type(image) == np.array:
        pass
        if image.shape[2] <= 3 and image.shape[0] >= 3:
            image = np.transpose(image, (2, 0, 1))
    elif type(image) == Image.Image:
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
    else:
        raise TypeError(f"image must be numpy array or PIL image and is {type(image)}")
    
    tiles = []
    for i in range(0, image.shape[1], tile_size):
        line_tiles = []
        for j in range(0, image.shape[2], tile_size):
            tile = image[:, i:i+tile_size, j:j+tile_size].copy()
            line_tiles.append(tile)
        tiles.append(np.array(line_tiles))

    return np.array(tiles)

def to_untiled(tiles):
    if type(tiles) == torch.Tensor:
        cat = torch.cat
    elif type(tiles) == np.array or type(tiles) == np.ndarray:
        cat = np.concatenate
    else:
        raise TypeError(f"tiles matrix must be tensor or numpy array and is {type(tiles)}")
    
    image = []
    for i in range(0, tiles.shape[0]):
        image_line = []
        for j in range(0, tiles.shape[1]):
            image_line.append(tiles[i,j])
        image.append(cat(image_line, 2))
    
    return cat(image, 1)
    
def show_image(image):
    if type(image) == np.ndarray or type(image) == np.array:
        if image.shape[0] <= 3 and image.shape[2] >= 3:
            image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image, "RGB")
    elif type(image) == Image.Image:
        pass
    else:
        raise TypeError(f"image must be tensor or PIL image and is {type(image)}")

    image.show()

def pad(pil_img, horiz, vert):
    width, height = pil_img.size
    new_width = width + 2 * horiz
    new_height = height + 2 * vert
    result = Image.new(pil_img.mode, (new_width, new_height), (0, 0, 0))
    result.paste(pil_img, (horiz, vert))
    return result

def mask_tile(tiles, coords):
    masked_tile = tiles[coords].copy()
    masked_array = tiles.copy()
    masked_array[coords] = np.zeros(masked_tile.shape)
    return masked_array, masked_tile