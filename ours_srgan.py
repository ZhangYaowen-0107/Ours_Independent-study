import tensorlayerx as tlx
from tensorlayerx.nn import Module
from tensorlayerx.nn import Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d, Flatten, Sequential
from tensorlayerx.nn import Linear, MaxPool2d

W_init = tlx.initializers.TruncatedNormal(stddev=0.02)
G_init = tlx.initializers.TruncatedNormal(mean=1.0, stddev=0.02)

class ResidualBlock(Module):
    def __init__(self,n):
        super(ResidualBlock, self).__init__()
        if n < 3:
            self.Inception = mini_inception()
        else:
            self.Inception = Inception_A()

    def forward(self, x):
        y = self.Inception(x)
        return y + x

class RRIM(Module):
    def __init__(self,n):
        super(RRIM, self).__init__()
        self.RB1 = ResidualBlock(n)
        self.RB2 = ResidualBlock(n)
        self.RB3 = ResidualBlock(n)
        
    def forward(self, x):
        y = self.RB1(x)
        y = self.RB2(y)
        y = self.RB3(y)
        return y + x

class Deep_layer(Module):
    def __init__(self):
        super(Deep_layer, self).__init__()
        self.conv1 = Conv2d(out_channels=64, kernel_size=(1,1), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)

        self.conv2_1 = Conv2d(out_channels=64, kernel_size=(1,1), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)
        self.conv2_2 = Conv2d(out_channels=48, kernel_size=(1,3), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)
        self.conv2_3 = Conv2d(out_channels=48, kernel_size=(3,1), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)

        self.conv3_1 = Conv2d(out_channels=64, kernel_size=(1,1), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)
        self.conv3_2 = Conv2d(out_channels=72, kernel_size=(1,3), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)
        self.conv3_3 = Conv2d(out_channels=80, kernel_size=(3,1), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)
        self.conv3_4 = Conv2d(out_channels=48, kernel_size=(1,3), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)
        self.conv3_5 = Conv2d(out_channels=48, kernel_size=(3,1), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)

        self.concat = tlx.nn.Concat(concat_dim=-1, name='concat_layer')

    def forward(self, x):
        y1 = self.conv1(x)

        y2 = self.conv2_1(x)
        y2_1 = self.conv2_2(y2)
        y2_2 = self.conv2_3(y2)

        y3 = self.conv3_1(x)
        y3 = self.conv3_2(y3)
        y3 = self.conv3_3(y3)
        y3_1 = self.conv3_4(y3)
        y3_2 = self.conv3_5(y3)

        x = self.concat([y1, y2_1, y2_2, y3_1, y3_2]) # 64 + 48 * 4 = 256 channel

        return x

class Inception_A(Module):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.conv1 = Conv2d(out_channels=64, kernel_size=(1,1), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)

        self.conv2_1 = Conv2d(out_channels=32, kernel_size=(1,1), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)
        self.conv2_2 = Conv2d(out_channels=64, kernel_size=(3,3), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)

        self.conv3_1 = Conv2d(out_channels=32, kernel_size=(1,1), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)
        self.conv3_2 = Conv2d(out_channels=64, kernel_size=(1,5), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)
        self.conv3_3 = Conv2d(out_channels=64, kernel_size=(5,1), stride=(1,1), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)

        self.concat = tlx.nn.Concat(concat_dim=-1, name='concat_layer')

    def forward(self, x):
        y1 = self.conv1(x)

        y2 = self.conv2_1(x)
        y2 = self.conv2_2(y2)

        y3 = self.conv3_1(x)
        y3 = self.conv3_2(y3)
        y3 = self.conv3_3(y3)

        x = self.concat([y1, y2, y3]) # 64 * 3 = 192 channel

        return x
    
class mini_inception(Module):
    def __init__(self):
        super(mini_inception, self).__init__()
        self.conv1_1 = Conv2d(out_channels=64, kernel_size=(1,1), stride=(1,1), act=None, padding='SAME', W_init=W_init)
        self.conv1_2 = Conv2d(out_channels=96, kernel_size=(3,3), stride=(1,1), act=None, padding='SAME', W_init=W_init)

        self.conv2_1 = Conv2d(out_channels=64, kernel_size=(1,1), stride=(1,1), act=None, padding='SAME', W_init=W_init)
        self.conv2_2 = Conv2d(out_channels=96, kernel_size=(1,7), stride=(1,1), act=None, padding='SAME', W_init=W_init)
        self.conv2_3 = Conv2d(out_channels=96, kernel_size=(7,1), stride=(1,1), act=None, padding='SAME', W_init=W_init)

        self.concat = tlx.nn.Concat(concat_dim=-1, name='concat_layer')

    def forward(self, x):
        y1 = self.conv1_1(x)
        y1 = self.conv1_2(y1)

        y2 = self.conv2_1(x)
        y2 = self.conv2_2(y2)
        y2 = self.conv2_3(y2)

        return self.concat([y1, y2]) + x # 96 * 2 = 192 channel

class SRGAN_g(Module):
    """ Generator in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
    feature maps (n) and stride (s) feature maps (n) and stride (s)
    """
    def __init__(self):
        super(SRGAN_g, self).__init__()
        self.conv1 = Conv2d(out_channels=64, kernel_size=(3,3), stride=(1,1), act=tlx.ReLU, padding='SAME', W_init=W_init)
        self.conv2 = Conv2d(out_channels=192, kernel_size=(3,3), stride=(1,1), act=tlx.ReLU, padding='SAME', W_init=W_init)

        self.residual_in_residual_block = self.make_layer()
        
        self.subpiexlconv1 = SubpixelConv2d(scale=2, act = tlx.ReLU)
        self.deep = Deep_layer()
        self.subpiexlconv2 = SubpixelConv2d(scale=2, act = tlx.ReLU)
        
        self.conv4 = Conv2d(3, kernel_size=(1,1), stride=(1,1), act=tlx.Tanh, padding='SAME', W_init=W_init)

    def make_layer(self):
        layer_list = []
        
        for i in range(4):
            layer_list.append(RRIM(i))
        return Sequential(layer_list)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        temp = x

        x = self.residual_in_residual_block(x)
        x = temp + x

        x = self.subpiexlconv1(x)
        x = self.deep(x)
        x = self.subpiexlconv2(x)
        
        x = self.conv4(x)
        
        return x

class SRGAN_d(Module):

    def __init__(self, dim = 64):
        super(SRGAN_d,self).__init__()
        self.conv1 = Conv2d(out_channels=dim, kernel_size=(4,4), stride=(2,2), act=tlx.LeakyReLU, padding='SAME', W_init=W_init)
        self.conv2 = Conv2d(out_channels=dim * 2, kernel_size=(4,4), stride=(2,2), act=None, padding='SAME', W_init=W_init, b_init=None)
        self.bn1 = BatchNorm2d(num_features=dim * 2, act=tlx.LeakyReLU, gamma_init=G_init)
        self.conv3 = Conv2d(out_channels=dim * 4, kernel_size=(4,4), stride=(2,2), act=None, padding='SAME', W_init=W_init, b_init=None)
        self.bn2 = BatchNorm2d(num_features=dim * 4,act=tlx.LeakyReLU,  gamma_init=G_init)
        self.conv4 = Conv2d(out_channels=dim * 8, kernel_size=(4, 4), stride=(2, 2), act=None, padding='SAME',W_init=W_init, b_init=None)
        self.bn3 = BatchNorm2d(num_features=dim * 8, act=tlx.LeakyReLU,  gamma_init=G_init)
        self.conv5 = Conv2d(out_channels=dim * 16, kernel_size=(4, 4), stride=(2, 2), act=None, padding='SAME',
                            W_init=W_init, b_init=None)
        self.bn4 = BatchNorm2d(num_features=dim * 16, act=tlx.LeakyReLU,  gamma_init=G_init)
        self.conv6 = Conv2d(out_channels=dim * 32, kernel_size=(4, 4), stride=(2, 2), act=None, padding='SAME',
                            W_init=W_init, b_init=None)
        self.bn5 = BatchNorm2d(num_features=dim * 32,act=tlx.LeakyReLU,  gamma_init=G_init)
        self.conv7 = Conv2d(out_channels=dim * 16, kernel_size=(1, 1), stride=(1, 1), act=None, padding='SAME',
                            W_init=W_init, b_init=None)
        self.bn6 = BatchNorm2d(num_features=dim * 16,act=tlx.LeakyReLU,  gamma_init=G_init)
        self.conv8 = Conv2d(out_channels=dim * 8, kernel_size=(1, 1), stride=(1, 1), act=None, padding='SAME',
                            W_init=W_init, b_init=None)
        self.bn7 = BatchNorm2d(num_features=dim * 8,act=None,  gamma_init=G_init)
        self.conv9 = Conv2d(out_channels=dim * 2, kernel_size=(1, 1), stride=(1, 1), act=None, padding='SAME',
                            W_init=W_init, b_init=None)
        self.bn8 = BatchNorm2d(num_features=dim * 2,act=tlx.LeakyReLU,  gamma_init=G_init)
        self.conv10 = Conv2d(out_channels=dim * 2, kernel_size=(3, 3), stride=(1, 1), act=None, padding='SAME',
                            W_init=W_init, b_init=None)
        self.bn9 = BatchNorm2d(num_features=dim * 2,act=tlx.LeakyReLU,  gamma_init=G_init)
        self.conv11 = Conv2d(out_channels=dim * 8, kernel_size=(3, 3), stride=(1, 1), act=None, padding='SAME',
                            W_init=W_init, b_init=None)
        self.bn10 = BatchNorm2d(num_features=dim * 8, gamma_init=G_init)
        self.add = Elementwise(combine_fn=tlx.add, act=tlx.LeakyReLU)
        self.flat = Flatten()
        self.dense = Linear(out_features=1,  W_init=W_init)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.conv6(x)
        x = self.bn5(x)
        x = self.conv7(x)
        x = self.bn6(x)
        x = self.conv8(x)
        x = self.bn7(x)
        temp = x
        x = self.conv9(x)
        x = self.bn8(x)
        x = self.conv10(x)
        x = self.bn9(x)
        x = self.conv11(x)
        x = self.bn10(x)
        x = self.add([temp, x])
        x = self.flat(x)
        x = self.dense(x)

        return x



class Vgg19_simple_api(Module):

    def __init__(self):
        super(Vgg19_simple_api,self).__init__()
        """ conv1 """
        self.conv1 = Conv2d(out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv2 = Conv2d(out_channels=64, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.maxpool1 = MaxPool2d(kernel_size=(2,2), stride=(2,2), padding='SAME')
        """ conv2 """
        self.conv3 = Conv2d(out_channels=128, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv4 = Conv2d(out_channels=128, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.maxpool2 = MaxPool2d(kernel_size=(2,2), stride=(2,2), padding='SAME')
        """ conv3 """
        self.conv5 = Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv6 = Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv7 = Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv8 = Conv2d(out_channels=256, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.maxpool3 = MaxPool2d(kernel_size=(2,2), stride=(2,2), padding='SAME')
        """ conv4 """
        self.conv9 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv10 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv11 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv12 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.maxpool4 = MaxPool2d(kernel_size=(2,2), stride=(2,2), padding='SAME') # (batch_size, 14, 14, 512)
        """ conv5 """
        self.conv13 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv14 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv15 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.conv16 = Conv2d(out_channels=512, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME')
        self.maxpool5 = MaxPool2d(kernel_size=(2,2), stride=(2,2), padding='SAME') # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        self.flat = Flatten()
        self.dense1 = Linear(out_features=4096, act=tlx.ReLU)
        self.dense2 = Linear(out_features=4096, act=tlx.ReLU)
        self.dense3 = Linear(out_features=1000, act=tlx.identity)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.maxpool3(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxpool4(x)
        conv = x
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.maxpool5(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x, conv