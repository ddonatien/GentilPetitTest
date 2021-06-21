# TODO separate in dicts
data_root = "/mnt/disk1/datasets/EPP"
tile_size = 64
max_epoch = 50
ann_file  = "/mnt/disk1/datasets/EPP/EPP.json"

nb_gpus = -1 # -1 for all availables
batch_size = 16
N = 3
num_workers = 16
train_ed = True
# conv_ed_file = '/mnt/disk1/project/GentilPetitTest/ConvED_June15_12-21-05_final.pth'
conv_ed_file = '/mnt/disk1/project/GentilPetitTest/VAE_June21_19-40-43_final.pth'