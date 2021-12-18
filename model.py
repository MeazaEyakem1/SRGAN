import math
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

       
        # new resolution 256*192

        #new resoulution 128 * 96
        self.conv1_e1 = ConvAct(1, 64)
        self.conv1_e2 = ConvAct(64, 64)
        self.down1 = nn.MaxPool2d(2, 2)

       
        #new resoulution 64 * 48
        self.conv2_e1 = ConvAct(64, 128)
        self.conv2_e2 = ConvAct(128, 128)
        self.down2 = nn.MaxPool2d(2, 2)
        
      
        # new resolution: 32 x 16
        self.conv3_e1 = ConvAct(128, 256)
        self.conv3_e2 = ConvAct(256, 256)
        self.down3 = nn.MaxPool2d(2, 2)
        
      
        # new resolution: 16 x 12
        self.conv4_e1 = ConvAct(256, 512)
        self.conv4_e2 = ConvAct(512, 512)
        self.down4 = nn.MaxPool2d(2, 2)

        # new resolution: 8 x 6
        self.conv5_1 = ConvAct(512, 1024)
        self.conv5_2 = ConvAct(1024, 1024) 
    
        #decoding part 
        # resolution: 16 x 12
        self.up4 = nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1)
        self.conv4_d1 = ConvAct(1024, 512)
        self.conv4_d2 = ConvAct(512, 512)

        # resolution: 32 x 24
        self.up3 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1)
        self.conv3_d1 = ConvAct(512, 256)
        self.conv3_d2 = ConvAct(256, 256)

        # resolution: 64 * 48
        self.up2 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.conv2_d1 = ConvAct(256, 128)
        self.conv2_d2 = ConvAct(128, 128)

        # resolution: 128 x 96
        self.up1 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.conv1_d1 = ConvAct(128, 64)
        self.conv1_d2 = ConvAct(64, 64)

        self.output_layer = ConvAct(64,1,1,1)
   

    def forward(self, x):
        
        # level 1 - encode
        x = self.conv1_e1(x)
        x = out1 = self.conv1_e2(x)
        x = self.down1(x)

        # level 2 - encode
        x = self.conv2_e1(x)
        x = out2 = self.conv2_e2(x)
        x = self.down2(x)

        # level 3 - encode
        x = self.conv3_e1(x)
        x = out3 = self.conv3_e2(x)
        x = self.down3(x)

        # level 4 - encode
        x = self.conv4_e1(x)
        x = out4 = self.conv4_e2(x)
        x = self.down4(x)

        # level 5 - encode/decode
        x = self.conv5_1(x)
        x = self.conv5_2(x)

        # level 4 - decode
        x = self.up4(x)
        x = torch.cat((x, out4), dim=1)
        x = self.conv4_d1(x)
        x = self.conv4_d2(x)

        # level 3 - decode
        x = self.up3(x)
        x = torch.cat((x, out3), dim=1)
        x = self.conv3_d1(x)
        x = self.conv3_d2(x)

        # level 2 - decode
        x = self.up2(x)
        x = torch.cat((x, out2), dim=1)
        x = self.conv2_d1(x)
        x = self.conv2_d2(x)

        # level 1 - decode
        x = self.up1(x)
        x = torch.cat((x,out1),dim=1)
        x = self.conv1_d1(x)
        x = self.conv1_d2(x)

        # make predictions
        x = self.output_layer(x)
        #x = lin(x)
        #x = x.view(x.size(0), 1, 256, 192)
        #print(shape(x))
        return x

    

_named_activations = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'prelu': nn.PReLU,
    'leakyrelu': nn.LeakyReLU,
    'elu': nn.ELU,
    'Tanh':nn.Tanh
}

def same_padding(kernel, dilation):
    return int((kernel + (kernel - 1) * (dilation - 1) - 1) / 2)


def get_activation_by_name(name, **kwargs):
    return _named_activations[name](**kwargs)


class ConvAct(nn.Module):
    """Conv->Activation"""
    def __init__(
            self, in_channels, out_channels,
            kernel=3, dilation=1, activation='relu'):
        super(ConvAct, self).__init__()
        padding = same_padding(kernel, dilation)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel,
            padding=padding,
            dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = get_activation_by_name(activation)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
      x = self.conv(x)
      x = self.bn(x)
      return self.activation(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
