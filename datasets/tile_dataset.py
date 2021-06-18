import os
import numpy as np
from random import randrange

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from pycocotools.coco import COCO

from modules.positionalEncoding import PositionalEncoding
from utils.imageTools import to_tiles, pad

class TileDataset(Dataset):
    def __init__(self, cfg, transform=None, load_to_ram=False):
        super().__init__()
        self.cfg = cfg
        self.coco = COCO(self.cfg.ann_file)
        self.root = cfg.data_root
        self.cat_names = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.pe = PositionalEncoding
        self.loaded = False
        if load_to_ram:
            # TODO Allow multiple workers ?
            self.images_db = {}
            self._load_all_images()
            self.loaded = True

    def __len__(self):
        return len(self.ids)

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_all_images(self):
        for id in self.ids:
            self.images_db[id] = self._load_image(id)

    def __getitem__(self, idx):
        ## Load image
        id = self.ids[idx]
        if self.loaded:
            image = self.images_db[id]
        else:
            image = self._load_image(id)
        width, height = image.size
        horiz_pad, vert_pad = int((self.cfg.tile_size - width % self.cfg.tile_size) / 2),\
                              int((self.cfg.tile_size - height % self.cfg.tile_size) / 2)
        image = pad(image, horiz_pad, vert_pad)
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
        tiles = to_tiles(image, self.cfg.tile_size)

        ## Single out random target
        target_idx = (randrange(tiles.shape[0]), randrange(tiles.shape[1]))

        target = tiles[target_idx].copy()
        tiles[target_idx] = np.zeros((3, self.cfg.tile_size, self.cfg.tile_size))

        return (torch.from_numpy(tiles), torch.from_numpy(target))
