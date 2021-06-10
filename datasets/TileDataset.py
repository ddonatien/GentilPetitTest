import os
import numpy as np
from random import randrange

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from pycocotools.coco import COCO

from utils.positionalEncoding import PositionalEncoding

class TileDataset(Dataset):
    def __init__(self, cfg, transform=None):
        super().__init__()
        self.cfg = cfg
        self.coco = COCO(self.cfg.ann_file)
        self.root = cfg.data_root
        self.cat_names = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.pe = PositionalEncoding

    def __len__(self):
        return len(self.ids)

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, idx):
        ## Load image
        id = self.ids[idx]
        image = self._load_image(id)
        width, height = image.size
        horiz_pad, vert_pad = int((self.cfg.tile_size - width % self.cfg.tile_size) / 2),\
                              int((self.cfg.tile_size - height % self.cfg.tile_size) / 2)
        image = self._pad(image, horiz_pad, vert_pad)
        # image.show()

        t = self.transform
        ## Do the transform
        if t == None:
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        image = t(image)

        ## Separate image in n tile_size sized tiles
        img_np = image.numpy()
        tiles = []
        for i in range(0, img_np.shape[1], self.cfg.tile_size):
            line_tiles = []
            for j in range(0, img_np.shape[2], self.cfg.tile_size):
                tile = img_np[:, i:i+self.cfg.tile_size, j:j+self.cfg.tile_size].copy()
                line_tiles.append(tile)
            tiles.append(np.array(line_tiles))
        tiles = np.array(tiles)
        # print(len(tiles))
        # img = Image.fromarray(tiles[225], 'RGB')
        # img.show()
        # print(img_np.shape)
        # print(type(img_np))

        ## Do positional encoding

        ## Single out random target
        target_idx = (randrange(tiles.shape[0]), randrange(tiles.shape[1]))

        target = tiles[target_idx].copy()
        tiles[target_idx] = np.zeros((3, self.cfg.tile_size, self.cfg.tile_size))

        return (torch.from_numpy(tiles), torch.from_numpy(target))

    def _pad(self, pil_img, horiz, vert):
        width, height = pil_img.size
        new_width = width + 2 * horiz
        new_height = height + 2 * vert
        result = Image.new(pil_img.mode, (new_width, new_height), (0, 0, 0))
        result.paste(pil_img, (horiz, vert))
        return result