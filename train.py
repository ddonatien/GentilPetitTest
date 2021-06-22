from modules.vanillaVAE import VanillaVAE
import os
import datetime
import torch
import time
from torch import nn
from torch.nn.modules import module
from torch.optim import Optimizer
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
import configs.test as cfg
from modules.losses import LogCoshLoss
from modules.convED import ConvED
from modules.encoderStack import  make_model
from datasets.tile_dataset import TileDataset
from utils.metering import AverageMeter

NB_GPUS = torch.cuda.device_count()
if cfg.nb_gpus >= 0:
    NB_GPUS = cfg.nb_gpus
BATCH_SIZE = cfg.batch_size
N = cfg.N
NUM_WORKERS = cfg.num_workers
TRAIN_ED = cfg.train_ed
CONV_ED_FILE = cfg.conv_ed_file

def run_epoch(data_iter, model, loss_compute, optimizer, device, epoch, loss_meter, writer, scheduler):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, data in enumerate(data_iter):
        data[0] = data[0].to(device)
        data[1] = data[1].to(device)
        out = model.forward(data[0])
        loss = loss_compute(model.module.generator(out), data[1])
        loss_meter.update(loss.item(), data[0].shape[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss
        total_tokens += data[0].shape[0]
        tokens += data[0].shape[0]
        if i % 2 == 1:
            elapsed = time.time() - start
            print("Epoch %d Step: %d Loss: %f Tokens per Sec: %f" %
                    (epoch, i, loss / data[0].shape[0], tokens / elapsed))
            writer.add_scalar('training loss',
                              epoch_losses.avg,
                              epoch * len(data_iter) + i)
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

class NoamOpt(Optimizer):
    "Optim wrapper that implements rate."
    def __init__(self, params, model_size, factor, warmup, optimizer, writer):
        super(NoamOpt, self).__init__(params, optimizer.defaults)
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        writer.add_scalar('lr',
                          self._rate,
                          self._step)
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model, writer):
    return NoamOpt(model.parameters(), model.module.d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            writer)

start_date = datetime.datetime.now().strftime("%B%d_%H-%M-%S")

dataset = TileDataset(cfg, load_to_ram=True)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log_cosh = LogCoshLoss()

if TRAIN_ED:
    # vae_model = VanillaVAE(3, 256)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     vae_model = nn.DataParallel(vae_model)
    # vae_model.to(device)

    # optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.005)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # for epoch in range(cfg.max_epoch):
    #     epoch_losses = AverageMeter()
    #     vae_model.train()
    #     with tqdm(total=32760) as _tqdm:
    #         _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, cfg.max_epoch))
    #         for tiles, target in data_loader:
    #             tiles = tiles.to(device)
    #             inputs = tiles.flatten(start_dim=0, end_dim=2)
    #             preds = vae_model(inputs)

    #             loss = vae_model.module.loss_function(*preds, M_N = 0.005)['loss']

    #             epoch_losses.update(loss, len(tiles))

    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             scheduler.step()
    #             _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
    #             _tqdm.update(len(inputs))

    # torch.save(vae_model.module.state_dict(), os.path.join('./', 'VAE_{}_final.pth'.format(start_date)))
    
    featuresED = ConvED()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        featuresED = nn.DataParallel(featuresED)
    featuresED.to(device)

    optimizer = torch.optim.Adam(featuresED.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    ## Train autoencoder
    for epoch in range(cfg.max_epoch):
        epoch_losses = AverageMeter()
        featuresED.train()
        # with tqdm(total=(len(dataset) - len(dataset) % BATCH_SIZE)) as _tqdm:
        with tqdm(total=32760) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, cfg.max_epoch))
            for tiles, target in data_loader:
                tiles = tiles.to(device)
                inputs = tiles.flatten(start_dim=0, end_dim=2)
                preds = featuresED(inputs)

                loss = log_cosh(preds, inputs)

                epoch_losses.update(loss, len (tiles))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))

    torch.save(featuresED.module.state_dict(), os.path.join('./', 'ConvED_{}_final.pth'.format(start_date)))

else:
    # featuresED.load_state_dict(torch.load(CONV_ED_FILE))
    featuresED = featuresED.to(device)
    featuresED.eval()
    writer = SummaryWriter()
    model = make_model(featuresED.encoder, featuresED.decoder, N=N)
    featuresED.load_state_dict(torch.load(CONV_ED_FILE))
    featuresED.requires_grad_(requires_grad=False)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = get_std_opt(model, writer)
    model.train()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    for epoch in range(cfg.max_epoch):
        epoch_losses = AverageMeter()
        lt = run_epoch(data_loader, model, log_cosh, optimizer, device, epoch, epoch_losses, writer, None)
        writer.add_scalar('training_loss/total_tokens',
                            lt,
                            (epoch + 1) * len(data_loader))
        if epoch % 100 == 1:
            torch.save(model.module.state_dict(), os.path.join('./', 'encoderStack_{}_epoch{}.pth'.format(start_date, epoch)))
    torch.save(model.module.state_dict(), os.path.join('./', 'encoderStack_{}_final.pth'.format(start_date)))

# inv_normalize = transforms.Normalize(
#    mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
#    std=[1/0.2023, 1/0.1994, 1/0.2010]
# )
# inv_tensor = inv_normalize(tensor)