import os
# Framework - tensorflow
os.environ['TL_BACKEND'] = 'tensorflow'
# use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import cv2

import tensorlayerx as tlx

from ours_srgan import SRGAN_g

# create folders to save result images and trained models
batch_size = 4
save_dir = "./media/output/"
tlx.files.exists_or_mkdir(save_dir)
checkpoint_dir = "./sr_image/SRGAN/models/"
tlx.files.exists_or_mkdir(checkpoint_dir)

G = SRGAN_g()
G.init_build(tlx.nn.Input(shape=(batch_size, 96, 96, 3)))

def evaluate(img_path):
    G.load_weights(os.path.join(checkpoint_dir, 'g.npz'), format='npz_dict')
    G.set_eval()

    hr_img = cv2.imread(img_path)
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
    lr_img = np.asarray(hr_img)
    lr_img_tensor = (lr_img / 127.5) - 1

    lr_img_tensor = np.asarray(lr_img_tensor, dtype=np.float32)
    lr_img_tensor = lr_img_tensor[np.newaxis, :, :, :]
    lr_img_tensor= tlx.ops.convert_to_tensor(lr_img_tensor)
    size = [lr_img.shape[0], lr_img.shape[1]]

    out = tlx.ops.convert_to_numpy(G(lr_img_tensor))
    out = np.asarray((out + 1) * 127.5, dtype=np.uint8)
    print("LR size: %s / Generated HR size: %s" % (size, out.shape))
    print("[*] save images")
    tlx.vision.save_image(out[0], file_name='gen.png', path=save_dir)
    out_cubic = cv2.resize(lr_img, dsize = [size[1] * 4, size[0] * 4], interpolation = cv2.INTER_CUBIC)
    tlx.vision.save_image(out_cubic, file_name='hr_cubic.png', path=save_dir)

# if run as main
if __name__ == '__main__':
    import argparse

    save_dir = "./output/"
    tlx.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "./models/"
    tlx.files.exists_or_mkdir(checkpoint_dir)

    parser = argparse.ArgumentParser()

    parser.add_argument('--img', type=str, default='img2.png')

    args = parser.parse_args()

    evaluate(args.img)
