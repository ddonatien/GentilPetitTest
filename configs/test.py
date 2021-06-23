# TODO separate in dicts
data_root = "/mnt/disk1/datasets/EPP"
tile_size = 64
max_epoch = 10000
ann_file  = "/mnt/disk1/datasets/EPP/EPP.json"

nb_gpus = -1 # -1 for all availables
batch_size = 64
N = 3
num_workers = 16
train_ed = False
# conv_ed_file = '/mnt/disk1/project/GentilPetitTest/ConvED_June22_20-21-15_final.pth'
conv_ed_file = '/mnt/disk1/project/GentilPetitTest/LeakyConvED_June23_17-01-27_final.pth'
# conv_ed_file = '/mnt/disk1/project/GentilPetitTest/VAE_June21_19-40-43_final.pth'