from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

config.TRAIN.n_epoch_init = 10
config.TRAIN.n_epoch = 400

config.TRAIN.hr_img_path = 'DIV2K/DIV2K_train_HR/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
