import sys
import torch
import numpy as np
import configs.test as cfg
from PIL import Image
from utils.imageTools import mask_tile, to_tiles, to_untiled, show_image, pad
from modules.convED import ConvED
from modules.encoderStack import  make_model
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

ENCODER_STACK_FILE = "/mnt/disk1/project/GentilPetitTest/encoderStack_June17_12-34-37_final.pth"
ENCODER_STACK_FILE = "/mnt/disk1/project/GentilPetitTest/encoderStack_June18_16-07-20_final.pth"
IMAGE_FILE = "/mnt/disk1/datasets/EPP/rgb018_Color.png"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
featuresED = ConvED()
featuresED.load_state_dict(torch.load(cfg.conv_ed_file))
featuresED = featuresED.to(device)
featuresED.eval()
model = make_model(featuresED.encoder, featuresED.decoder, N=cfg.N)
model.load_state_dict(torch.load(ENCODER_STACK_FILE))
model.to(device)
model.eval()

featuresED.load_state_dict(torch.load(cfg.conv_ed_file))

# for p in model.parameters():
#     print(p)

t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)),
])

inv_t = transforms.Compose([
    transforms.Normalize(
        (-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010),
        (1/0.2023, 1/0.1994, 1/0.2010)),
    transforms.ToPILImage()
])

image = Image.open(IMAGE_FILE).convert("RGB")
width, height = image.size
horiz_pad, vert_pad = int((cfg.tile_size - width % cfg.tile_size) / 2),\
                        int((cfg.tile_size - height % cfg.tile_size) / 2)
image = pad(image, horiz_pad, vert_pad)

## Pure numpy 
# tiles = to_tiles(image, cfg.tile_size)
# show_image(to_untiled(tiles))

## Torch tensor
image = t(image)
tiles = torch.tensor(to_tiles(image.numpy(), cfg.tile_size))
show_image(inv_t(to_untiled(tiles)))

generated_tiles = []
with torch.no_grad():
    for i in range(0, tiles.shape[0]):
        generated_line = []
        generated_line2 = []
        for j in range(0, tiles.shape[1]):
            masked_array, masked_tile = mask_tile(tiles.numpy(), (i, j))
            pred = model(torch.tensor(masked_array).unsqueeze(0).to(device))
            gen = model.generator(pred)
            generated_line.append(np.transpose(np.array(inv_t(gen[0])), (2, 0, 1)))
        generated_tiles.append(np.array(generated_line))

generated_image = to_untiled(np.array(generated_tiles))

show_image(generated_image)

# model = ConvED()
# model.load_state_dict(torch.load(cfg.conv_ed_file))
model = featuresED
# model.load_state_dict(torch.load(cfg.conv_ed_file))
model = model.to(device)
model.eval()

def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()

with torch.no_grad():

    pred = model.decoder(model.encoder(tiles.flatten(0, 1).to(device)))
    show(make_grid(pred, padding=2))

# dbg(untiled.shape)
# 
# show_image(untiled)