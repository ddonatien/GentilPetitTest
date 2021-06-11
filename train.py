import os
import datetime
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from tqdm import tqdm
import configs.test as cfg
from modules.losses import LogCoshLoss
from modules.encoder_decoder import ConvED
from datasets.tile_dataset import TileDataset
from utils.metering import AverageMeter

NB_GPUS = 2
BATCH_SIZE = 8 * NB_GPUS
start_date = datetime.datetime.now().strftime("%B%d_%H-%M-%S")

dataset = TileDataset(cfg)
a = dataset.__getitem__(0)
print(a[0].shape, a[1].shape)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log_cosh = LogCoshLoss()

model = ConvED()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)

## Train autoencoder
for epoch in range(cfg.max_epoch):
    epoch_losses = AverageMeter()
    model.train()
    with tqdm(total=(len(dataset) - len(dataset) % BATCH_SIZE)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, cfg.max_epoch))
        for tiles, target in data_loader:
            tiles.to(device)
            inputs = tiles.flatten(start_dim=0, end_dim=1)
            preds = model(inputs)

            loss = log_cosh(preds, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            _tqdm.update(len(inputs))

torch.save(model.module.state_dict(), os.path.join('./', 'ConvED_{}_final.pth'.format(start_date)))

inv_normalize = transforms.Normalize(
   mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
   std=[1/0.2023, 1/0.1994, 1/0.2010]
)
# inv_tensor = inv_normalize(tensor)
