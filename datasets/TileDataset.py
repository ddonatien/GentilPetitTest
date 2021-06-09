from torch.utils.data import Dataset

class TileDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.coco = COCO(self.cfg.ann_file)
        self.cat_names = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]

    def __len__(self):
        # TODO get from coco
        return 130

    def __getitem__(self, idx):
        # TODO get tiles

