import torch
import numpy as np
import configs.test as cfg
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from modules.convED import ConvED, LeakyConvED
from modules.vanillaVAE import VanillaVAE
from datasets.tile_dataset import TileDataset
from modules.losses import LogCoshLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = TileDataset(cfg)

# data_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=8,
#     pin_memory=True
# )

# model = VanillaVAE(3, 256)

model = LeakyConvED()
model.load_state_dict(torch.load(cfg.conv_ed_file))
model = model.to(device)
model.eval()

log_cosh = LogCoshLoss()

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

with torch.no_grad():
    tiles = dataset[0][0].flatten(start_dim=0, end_dim=1)
    tiles = tiles.to(device)
    print(tiles.shape)
    preds = model(tiles)
    # pred = preds[0]
    pred = preds
    print(pred.shape)
    # loss = model.loss_function(*preds, M_N = 0.005)['loss'].mean()
    loss= log_cosh(pred, tiles)
    print(f"loss = {loss}")
    show(make_grid(pred, padding=2))
    # plt.imshow(pred[0].cpu().permute(1, 2, 0))