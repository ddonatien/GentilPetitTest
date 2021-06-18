import numpy as np

def to_tiles(image, tile_size):
    img_np = image.numpy()
    tiles = []
    for i in range(0, img_np.shape[1], tile_size):
        line_tiles = []
        for j in range(0, img_np.shape[2], tile_size):
            tile = img_np[:, i:i+tile_size, j:j+tile_size].copy()
            line_tiles.append(tile)
        tiles.append(np.array(line_tiles))
    tiles = np.array(tiles)