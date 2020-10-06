import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.nn.init import orthogonal_


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        orthogonal_(m.weight)
        m.bias.data.fill_(0.)

def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def snconvtrans2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel, size=1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, size, size))

    def forward(self, image):
        one_channel = image[:,0:1,:,:]
        noise = one_channel.new(one_channel.shape).normal_(0, 1)
        out = image + self.weight * noise
        return out

class GenIniBlock(nn.Module):
    def __init__(self, z_dim, out_channels, size=1, add_noise=True):
        super().__init__()
        self.out_channels = out_channels
        self.add_noise = add_noise
        self.snlinear0 = snlinear(in_features=z_dim, out_features=out_channels * 4 * 4)
        if add_noise:
            self.noise0 = NoiseInjection(out_channels, size)

    def forward(self, z):
        act0 = self.snlinear0(z)
        act0 = act0.view(-1, self.out_channels, 4, 4)
        if self.add_noise:
            act0 = self.noise0(act0)

        return act0

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size=1, add_noise=True):
        super().__init__()
        self.conv_1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.add_noise = add_noise
        if add_noise:
            self.noise1 = NoiseInjection(out_channels, size)
            self.noise2 = NoiseInjection(out_channels, size)
        self.conv_0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')

    def forward(self, x,):
        x0 = x

        x = self.relu(self.bn1(x))
        x = self.upsample(x)  # upsample
        x = self.conv_1(x)
        if self.add_noise:
            x = self.noise1(x)
        x = self.relu(self.bn2(x))
        x = self.conv_2(x)
        if self.add_noise:
            x = self.noise2(x)
        x0 = self.upsample(x0)  # upsample
        x0 = self.conv_0(x0)

        out = x + x0
        return out

class Generator(nn.Module):
    r'''SAGAN Generator

    Args:
        latent_dim: latent dimension
        conv_dim: base number of channels
        image_size: image resolution
        out_channels: number of output channels
        add_noise: whether to add noises to each conv layer
        attn: whether to add self-attention layer
    '''

    def __init__(self, latent_dim, conv_dim=32, image_size=128, out_channels=3, add_noise=True, attn=True):
        super().__init__()

        self.latent_dim = latent_dim
        self.conv_dim = conv_dim
        self.image_size = image_size
        self.add_noise = add_noise
        self.attn = attn

        self.block0 = GenIniBlock(latent_dim, conv_dim * 16, 4, add_noise=add_noise)
        self.block1 = GenBlock(conv_dim * 16, conv_dim * 16, size=8, add_noise=add_noise)
        self.block2 = GenBlock(conv_dim * 16, conv_dim * 8, size=16, add_noise=add_noise)
        if image_size == 64:
            self.block3 = GenBlock(conv_dim * 8, conv_dim * 4, size=32, add_noise=add_noise)
            if attn:
                self.self_attn1 = Self_Attn(conv_dim * 4)
            self.block4 = GenBlock(conv_dim * 4, conv_dim * 2, size=64, add_noise=add_noise)
            conv_dim = conv_dim * 2
        elif image_size == 128:
            self.block3 = GenBlock(conv_dim * 8, conv_dim * 4, add_noise=add_noise)
            if attn:
                self.self_attn1 = Self_Attn(conv_dim * 4)
            self.block4 = GenBlock(conv_dim * 4, conv_dim * 2, add_noise=add_noise)
            # self.self_attn2 = Self_Attn(conv_dim*2)
            self.block5 = GenBlock(conv_dim * 2, conv_dim, add_noise=add_noise)
        else: # image_size == 256 or 512
            self.block3 = GenBlock(conv_dim * 8, conv_dim * 8, add_noise=add_noise)
            self.block4 = GenBlock(conv_dim * 8, conv_dim * 4, add_noise=add_noise)
            if attn:
                self.self_attn1 = Self_Attn(conv_dim * 4)
            self.block5 = GenBlock(conv_dim * 4, conv_dim * 2, add_noise=add_noise)
            self.block6 = GenBlock(conv_dim * 2, conv_dim, add_noise=add_noise)
            if image_size == 512:
                self.block7 = GenBlock(conv_dim, conv_dim, add_noise=add_noise)

        self.bn = nn.BatchNorm2d(conv_dim, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.toRGB = snconv2d(in_channels=conv_dim, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Weight init
        self.apply(init_weights)

    def forward(self, z):
        out = self.block0(z)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        if self.attn:
            out = self.self_attn1(out)
        out = self.block4(out)
        if self.image_size > 64:
            out = self.block5(out)
            if self.image_size == 256 or self.image_size == 512:
                out = self.block6(out)
                if self.image_size == 512:
                    out = self.block7(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.toRGB(out)
        out = self.tanh(out)
        return out


class DiscOptBlock(nn.Module):
    # Compared with block, optimized_block always downsamples the spatial resolution of the input vector by a factor of 4.
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x

        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.conv_0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.conv_0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, conv_dim, image_size=128, in_channels=3, out_channels=1, out_feature=False):
        super().__init__()
        self.conv_dim = conv_dim
        self.image_size = image_size
        self.out_feature = out_feature

        self.fromRGB = snconv2d(in_channels, conv_dim, 1, bias=True)

        self.block1 = DiscBlock(conv_dim, conv_dim * 2)
        self.self_attn = Self_Attn(conv_dim*2)
        self.block2 = DiscBlock(conv_dim * 2, conv_dim * 4)
        self.block3 = DiscBlock(conv_dim * 4, conv_dim * 8)
        if image_size == 64:
            self.block4 = DiscBlock(conv_dim * 8, conv_dim * 16)
            self.block5 = DiscBlock(conv_dim * 16, conv_dim * 16)
        elif image_size == 128:
            self.block4 = DiscBlock(conv_dim * 8, conv_dim * 16)
            self.block5 = DiscBlock(conv_dim * 16, conv_dim * 16)
            self.block6 = DiscBlock(conv_dim * 16, conv_dim * 16)
        else:
            self.block4 = DiscBlock(conv_dim * 8, conv_dim * 8)
            self.block5 = DiscBlock(conv_dim * 8, conv_dim * 16)
            self.block6 = DiscBlock(conv_dim * 16, conv_dim * 16)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=conv_dim*16, out_features=out_channels)

        # Weight init
        self.apply(init_weights)

    def forward(self, x):
        h0 = self.fromRGB(x)
        h1 = self.block1(h0)
        h1 = self.self_attn(h1)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        if self.image_size == 64:
            h5 = self.block5(h4, downsample=False)
            h6 = h5
        elif self.image_size == 128:
            h5 = self.block5(h4)
            h6 = self.block6(h5, downsample=False)
        else:
            h5 = self.block5(h4)
            h6 = self.block6(h5)
            h6 = self.block7(h6, downsample=False)
        h6 = self.relu(h6)

        # Global sum pooling
        h7 = torch.sum(h6, dim=[2,3])
        out = torch.squeeze(self.snlinear1(h7))

        if self.out_feature:
            return out, h7
        else:
            return out


class DisFinalBlock(nn.Module):
    """ Final block for the Discriminator """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_1 = snconv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv_2 = snconv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4)
        self.conv_3 = snconv2d(in_channels=in_channels, out_channels=1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.conv_1(x))
        y = self.relu(self.conv_2(y))
        y = self.conv_3(y)

        return y


class SNResMLPBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Sequential(
            snlinear(channels, channels),
            nn.ReLU(inplace=True)
        )
        self.fc2 = snlinear(channels, channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.fc2(self.fc1(x))
        out += x
        return self.relu(out)


class Discriminator_MLP(nn.Module):

    def __init__(self, in_channels, out_channels, out_feature=True, num_block=3):
        super().__init__()
        self.out_feature = out_feature
        self.num_block = num_block

        self.fc1 = nn.Sequential(
            snlinear(in_channels, out_channels),
            nn.ReLU(inplace=True)
        )
        self.block1 = SNResMLPBlock(out_channels)
        if num_block > 1:
            self.block2 = SNResMLPBlock(out_channels)
        if num_block > 2:
            self.block3 = SNResMLPBlock(out_channels)
        self.fc4 = snlinear(out_channels, 1)


        self.apply(init_weights)

    def forward(self, z):

        out = self.fc1(z)
        f = self.block1(out)
        if self.num_block > 1:
            f = self.block2(f)
        if self.num_block > 2:
            f = self.block3(f)
        out = self.fc4(f)

        if self.out_feature:
            return out, f
        else:
            return out


class DCDiscriminator(nn.Module):
    def __init__(self, conv_dim=64, image_size=64, image_channel=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_channel, conv_dim, 5, 2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(conv_dim, conv_dim * 2, 5, 2, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(conv_dim * 2, conv_dim * 4, 5, 2, 2),
            nn.LeakyReLU(inplace=True),
        )
        self.fc = nn.Linear(conv_dim * 4 * 4 * 4, 1)

    def forward(self, x):
        x = self.conv(x).view(x.size(0), -1)
        return self.fc(x)
