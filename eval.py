import torch
import numpy as np
import configs.test as cfg
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from modules.encoder_decoder import ConvED
from datasets.tile_dataset import TileDataset
from modules.losses import LogCoshLoss

MODEL_FILE = '/mnt/disk1/project/GentilPetitTest/ConvED_June15_12-21-05_final.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = TileDataset(cfg)

# data_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=8,
#     pin_memory=True
# )

model = ConvED()
model.load_state_dict(torch.load(MODEL_FILE))
print(model.encoder)
model = model.to(device)
model.eval()

log_cosh = LogCoshLoss()

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

with torch.no_grad():
    tiles = dataset[0][0].flatten(start_dim=0, end_dim=1)
    tiles = tiles.to(device)
    print(tiles.shape)
    pred = model.decoder(model.encoder(tiles))
    print(pred.shape)
    print(f"loss = {log_cosh(pred, tiles)}")
    show(make_grid(pred, padding=100))
    # plt.imshow(pred[0].cpu().permute(1, 2, 0))
    plt.show()