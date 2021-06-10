import configs.test as cfg
from datasets.TileDataset import TileDataset

dataset = TileDataset(cfg)
a = dataset.__getitem__(0)
print(a[0].shape, a[1].shape)
