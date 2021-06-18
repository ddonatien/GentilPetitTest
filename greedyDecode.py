import sys
import torch
from pydbg import dbg
import numpy as np
import configs.test as cfg
from PIL import Image
from utils.imageTools import mask_tile, to_tiles, to_untiled, show_image, pad
from modules.convED import ConvED
from modules.encoderStack import  make_model
from torchvision import transforms

ENCODER_STACK_FILE = "/mnt/disk1/project/GentilPetitTest/encoderStack_June17_12-34-37_final.pth"
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
generated_tiles2 = []
with torch.no_grad():
    for i in range(0, tiles.shape[0]):
        generated_line = []
        generated_line2 = []
        for j in range(0, tiles.shape[1]):
            masked_array, masked_tile = mask_tile(tiles.numpy(), (i, j))
            pred1 = model(torch.tensor(masked_array).unsqueeze(0).to(device))
            gen = model.generator(pred1)
            pred2 = featuresED(torch.tensor(masked_tile).unsqueeze(0).to(device))
            generated_line.append(np.transpose(np.array(inv_t(gen[0])), (2, 0, 1)))
            generated_line2.append(np.transpose(np.array(inv_t(pred2[0])), (2, 0, 1)))
        generated_tiles.append(np.array(generated_line))
        generated_tiles2.append(np.array(generated_line2))

generated_image = to_untiled(np.array(generated_tiles))
generated_image2 = to_untiled(np.array(generated_tiles2))

show_image(generated_image)
show_image(generated_image2)

# dbg(untiled.shape)
# 
# show_image(untiled)