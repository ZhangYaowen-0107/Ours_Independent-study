import os
os.environ['TL_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import tensorlayerx as tlx
from tensorlayerx.dataflow import Dataset, DataLoader
from ours_srgan import SRGAN_g, SRGAN_d

from config import config
from tensorlayerx.vision.transforms import Compose, RandomCrop, Normalize, RandomFlipHorizontal, Resize
import vgg
from tensorlayerx.model import TrainOneStep
from tensorlayerx.nn import Module
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU') 
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
###====================== HYPER-PARAMETERS ===========================###
batch_size = 4 # 8
n_epoch_init = config.TRAIN.n_epoch_init
n_epoch = config.TRAIN.n_epoch

checkpoint_dir = "models"
tlx.files.exists_or_mkdir(checkpoint_dir)

hr_transform = Compose([
    RandomCrop(size=(384, 384)),
    RandomFlipHorizontal(),
])

nor = Normalize(mean=(127.5), std=(127.5), data_format='HWC')
lr_transform = Resize(size=(96, 96))

train_hr_imgs = tlx.vision.load_images(path=config.TRAIN.hr_img_path, n_threads = 32)

class TrainData(Dataset):

    def __init__(self, hr_trans=hr_transform, lr_trans=lr_transform):
        self.train_hr_imgs = train_hr_imgs
        self.hr_trans = hr_trans
        self.lr_trans = lr_trans

    def __getitem__(self, index):
        img = self.train_hr_imgs[index]
        hr_patch = self.hr_trans(img)
        lr_patch = self.lr_trans(hr_patch)
        return nor(lr_patch), nor(hr_patch)

    def __len__(self):
        return len(self.train_hr_imgs)


class WithLoss_init(Module):
    def __init__(self, G_net, loss_fn):
        super(WithLoss_init, self).__init__()
        self.net = G_net
        self.loss_fn = loss_fn

    def forward(self, lr, hr):
        out = self.net(lr)
        loss = self.loss_fn(out, hr)

        return loss


class WithLoss_D(Module):
    def __init__(self, D_net, G_net, loss_fn):
        super(WithLoss_D, self).__init__()
        self.D_net = D_net
        self.G_net = G_net
        self.loss_fn = loss_fn

    def forward(self, lr, hr):
        fake_patchs = self.G_net(lr)
        logits_fake = self.D_net(fake_patchs)
        logits_real = self.D_net(hr)
        
        d_loss1 = self.loss_fn(logits_real, tlx.ones_like(logits_real))
        d_loss1 = tlx.ops.reduce_mean(d_loss1)

        d_loss2 = self.loss_fn(logits_fake, tlx.zeros_like(logits_fake))
        d_loss2 = tlx.ops.reduce_mean(d_loss2)

        d_loss = d_loss1 + d_loss2
        return d_loss


class WithLoss_G(Module):
    def __init__(self, D_net, G_net, vgg, loss_fn1, loss_fn2):
        super(WithLoss_G, self).__init__()
        self.D_net = D_net
        self.G_net = G_net
        self.vgg = vgg
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2

    def forward(self, lr, hr):
        fake_patchs = self.G_net(lr)
        logits_fake = self.D_net(fake_patchs)

        feature_fake = self.vgg((fake_patchs + 1) / 2.)
        feature_real = self.vgg((hr + 1) / 2.)

        g_gan_loss = 1e-3 * self.loss_fn1(logits_fake, tlx.ones_like(logits_fake))
        g_gan_loss = tlx.ops.reduce_mean(g_gan_loss)
        mse_loss = self.loss_fn2(fake_patchs, hr)
        vgg_loss = 2e-6 * self.loss_fn2(feature_fake, feature_real)

        g_loss = mse_loss + vgg_loss + g_gan_loss
        return g_loss

G = SRGAN_g()
D = SRGAN_d()
VGG = vgg.VGG19(pretrained=False, end_with='pool4', mode='dynamic')

G.init_build(tlx.nn.Input(shape=(batch_size, 96, 96, 3)))
D.init_build(tlx.nn.Input(shape=(batch_size, 384, 384, 3)))


def train():
    G.set_train()
    D.set_train()
    VGG.set_eval() 
    
    train_ds = TrainData()
    train_ds_img_nums = len(train_ds)
    train_ds = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    g_lr_v = tlx.optimizers.lr.StepDecay(learning_rate=0.03, step_size=1000, gamma=0.1, last_epoch=-1, verbose=True)
    d_lr_v = tlx.optimizers.lr.StepDecay(learning_rate=0.03, step_size=1000, gamma=0.1, last_epoch=-1, verbose=True)

    g_optimizer_init = tlx.optimizers.Momentum(g_lr_v, 0.9)
    g_optimizer = tlx.optimizers.Momentum(g_lr_v, 0.9)
    d_optimizer = tlx.optimizers.Momentum(d_lr_v, 0.9)

    g_weights = G.trainable_weights
    d_weights = D.trainable_weights
    
    net_with_loss_init = WithLoss_init(G, loss_fn=tlx.losses.mean_squared_error)
    net_with_loss_D = WithLoss_D(D_net=D, G_net=G, loss_fn=tlx.losses.sigmoid_cross_entropy)
    net_with_loss_G = WithLoss_G(D_net=D, G_net=G, vgg=VGG, loss_fn1=tlx.losses.sigmoid_cross_entropy,
                                 loss_fn2=tlx.losses.mean_squared_error)

    trainforinit = TrainOneStep(net_with_loss_init, optimizer=g_optimizer_init, train_weights=g_weights)
    trainforG = TrainOneStep(net_with_loss_G, optimizer=g_optimizer, train_weights=g_weights)
    trainforD = TrainOneStep(net_with_loss_D, optimizer=d_optimizer, train_weights=d_weights)

    n_step_epoch = round(train_ds_img_nums // batch_size)

    timeStart = time.time()

    for epoch in range(n_epoch_init):
        for step, (lr_patch, hr_patch) in enumerate(train_ds):
            step_time = time.time()
            loss = trainforinit(lr_patch, hr_patch)
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, float(loss)))
            f.write("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} \n".format(
                epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, float(loss)))
    
    G.save_weights(os.path.join(checkpoint_dir, 'g.npz'), format='npz_dict')
    D.save_weights(os.path.join(checkpoint_dir, 'd.npz'), format='npz_dict')
    f.write("\n[PreTrain] epoch: {}, StartTime: {}, EndTime: {}, Total: {:.3f}s\n\n".format(n_epoch_init, time.ctime(timeStart), time.ctime(), time.time() - timeStart))

    n_step_epoch = round(train_ds_img_nums // batch_size)

    start = time.time()
    for epoch in range(n_epoch):

        timeStart = time.time()

        for step, (lr_patch, hr_patch) in enumerate(train_ds):
            step_time = time.time()
            loss_g = trainforG(lr_patch, hr_patch)
            loss_d = trainforD(lr_patch, hr_patch)
            print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f}, d_loss: {:.3f} ".format(
                    epoch, n_epoch, step, n_step_epoch, time.time() - step_time, float(loss_g), float(loss_d)))
            f.write("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f}, d_loss: {:.3f} \n".format(
                    epoch, n_epoch, step, n_step_epoch, time.time() - step_time, float(loss_g), float(loss_d)))

        g_lr_v.step()
        d_lr_v.step()
        
        G.save_weights(os.path.join(checkpoint_dir, 'g.npz'), format='npz_dict')
        D.save_weights(os.path.join(checkpoint_dir, 'd.npz'), format='npz_dict')
        f.write("[Train]epoch{}, StartTime: {}, EndTime: {}, Total: {:.3f}s\n\n".format(epoch, time.ctime(timeStart), time.ctime(), time.time() - timeStart))
    f.write("[Train] End, StartTime: {}, EndTime: {}, Total: {:.3f}s\n\n".format(time.ctime(start), time.ctime(), time.time() - start))

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=str, default='train', help='train, re')

    args = parser.parse_args()
    tlx.global_flag['mode'] = args.m

    f = open("log.txt", 'a')
    f.write("\nRun time : {}\n".format(time.ctime()))

    if tlx.global_flag['mode'] == 're':
        n_epoch_init = 0
        G.load_weights(os.path.join(checkpoint_dir, 'g.npz'), format='npz_dict')
        D.load_weights(os.path.join(checkpoint_dir, 'd.npz'), format='npz_dict')
    
    train()
    
    f.close()
    
