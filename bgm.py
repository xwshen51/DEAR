from sagan import *
import torchvision.models as models
from resnet import *
import torch.nn.init as init
from causal_model import *


class ResEncoder(nn.Module):
    r'''ResNet Encoder

    Args:
        latent_dim: latent dimension
        arch: network architecture. Choices: resnet - resnet50, resnet18
        dist: encoder distribution. Choices: deterministic, gaussian, implicit
        fc_size: number of nodes in each fc layer
        noise_dim: dimension of input noise when an implicit encoder is used
    '''
    def __init__(self, latent_dim=64, arch='resnet', dist='gaussian', fc_size=2048, noise_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.dist = dist
        self.noise_dim = noise_dim

        in_channels = noise_dim + 3 if dist == 'implicit' else 3
        out_dim = latent_dim * 2 if dist == 'gaussian' else latent_dim
        if arch == 'resnet':
            self.encoder = resnet50(pretrained=False, in_channels=in_channels, fc_size=fc_size, out_dim=out_dim)
        else:
            assert arch == 'resnet18'
            self.encoder = resnet18(pretrained=False, in_channels=in_channels, fc_size=fc_size, out_dim=out_dim)

    def forward(self, x, avepool=False):
        '''
        :param x: input image
        :param avepool: whether to return the average pooling feature (used for downstream tasks)
        :return:
        '''
        if self.dist == 'implicit':
            # Concatenate noise with the input image x
            noise = x.new(x.size(0), self.noise_dim, 1, 1).normal_(0, 1)
            noise = noise.expand(x.size(0), self.noise_dim, x.size(2), x.size(3))
            x = torch.cat([x, noise], dim=1)
        z, ap = self.encoder(x)
        if avepool:
            return ap
        if self.dist == 'gaussian':
            return z.chunk(2, dim=1)
        else:
            return z


class BigDecoder(nn.Module):
    r'''Big generator based on SAGAN

    Args:
        latent_dim: latent dimension
        conv_dim: base number of channels
        image_size: image resolution
        dist: generator distribution. Choices: deterministic, gaussian, implicit
        g_std: scaling the standard deviation of the gaussian generator. Default: 1
    '''
    def __init__(self, latent_dim=64, conv_dim=32, image_size=64, dist='deterministic', g_std=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.dist = dist
        self.g_std = g_std

        out_channels = 6 if dist == 'gaussian' else 3
        add_noise = True if dist == 'implicit' else False
        self.decoder = Generator(latent_dim, conv_dim, image_size, out_channels, add_noise)

    def forward(self, z, mean=False, stats=False):
        out = self.decoder(z)
        if self.dist == 'gaussian':
            x_mu, x_logvar = out.chunk(2, dim=1)
            if stats:
                return x_mu, x_logvar
            else:
                x_sample = reparameterize(x_mu, (x_logvar / 2).exp(), self.g_std)
                if mean:
                    return x_mu
                else:
                    return x_sample
        else:
            return out


class BGM(nn.Module):
    r'''Bidirectional generative model

        Args:
            General settings:
                latent_dim: latent dimension
                conv_dim: base number of channels
                image_size: image resolution
                image_channel: number of image channel
            Encoder settings:
                enc_dist: encoder distribution
                enc_arch: encoder architecture
                enc_fc_size: number of nodes in each fc layer in encoder
                enc_noise_dim: dimension of input noise when an implicit encoder is used
            Generator settings:
                dec_dist: generator distribution. Choices: deterministic, implicit
                dec_arch: generator architecture. Choices: sagan, dcgan

    '''
    def __init__(self, latent_dim=64, conv_dim=32, image_size=64,
                 enc_dist='gaussian', enc_arch='resnet', enc_fc_size=2048, enc_noise_dim=128, dec_dist='implicit',
                 prior='gaussian', num_label=None, A=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.enc_dist = enc_dist
        self.dec_dist = dec_dist
        self.prior_dist = prior
        self.num_label = num_label

        self.encoder = ResEncoder(latent_dim, enc_arch, enc_dist, enc_fc_size, enc_noise_dim)
        self.decoder = BigDecoder(latent_dim, conv_dim, image_size, dec_dist)
        if 'scm' in prior:
            self.prior = SCM(num_label, A, scm_type=prior)

    def encode(self, x, mean=False, avepool=False):
        if avepool:
            return self.encoder(x, avepool=True)
        else:
            if self.enc_dist == 'gaussian':
                z_mu, z_logvar = self.encoder(x)
                if mean: # for downstream tasks
                    return z_mu
                else:
                    z_fake = reparameterize(z_mu, (z_logvar / 2).exp())
                    return z_fake
            else:
                return self.encoder(x)

    def decode(self, z, mean=True):
        if self.decoder_type != 'gaussian':
            return self.decoder(z)
        else: #gaussian
            return self.decoder(z, mu=mean)

    def traverse(self, eps, gap=3, n=10):
        dim = self.num_label if self.num_label is not None else self.latent_dim
        sample = torch.zeros((n * dim, 3, self.image_size, self.image_size))
        eps = eps.expand(n, self.latent_dim)
        if self.prior_dist == 'gaussian' or self.prior_dist == 'uniform':
            z = eps
        else:
            label_z = self.prior(eps[:, :dim])
            other_z = eps[:, dim:]
            z = torch.cat([label_z, other_z], dim=1)
        for idx in range(dim):
            traversals = torch.linspace(-gap, gap, steps=n)
            z_new = z.clone()
            z_new[:, idx] = traversals
            with torch.no_grad():
                sample[n * idx:(n * (idx + 1)), :, :, :] = self.decoder(z_new)
        return sample

    def forward(self, x=None, z=None, recon=False, infer_mean=True):
        # recon_mean is used for gaussian decoder which we do not use here.
        # Training Mode
        if x is not None and z is not None:
            if self.enc_dist == 'gaussian':
                z_mu, z_logvar = self.encoder(x)
                z_fake = reparameterize(z_mu, (z_logvar / 2).exp())
            else: # deterministic or implicit
                z_fake = self.encoder(x)

            if 'scm' in self.prior_dist:
                # in prior
                label_z = self.prior(z[:, :self.num_label]) # z after causal layer
                other_z = z[:, self.num_label:]
                z = torch.cat([label_z, other_z], dim=1)

            x_fake = self.decoder(z)

            if 'scm' in self.prior_dist:
                if self.enc_dist == 'gaussian' and infer_mean:
                    return z_fake, x_fake, z, z_mu
                else:
                    return z_fake, x_fake, z, None
            return z_fake, x_fake, z_mu

        # Inference Mode
        elif x is not None and z is None:
            # Get latent
            if self.enc_dist == 'gaussian':
                z_mu, z_logvar = self.encoder(x)
                z_fake = reparameterize(z_mu, (z_logvar / 2).exp())
            else: # deterministic or implicit
                z_fake = self.encoder(x)
            # Reconstruction
            if recon:
                return self.decoder(z_fake)

            # Representation
            # Mean representation for Gaussian encoder
            elif infer_mean and self.enc_dist == 'gaussian':
                return z_mu
            # Random representation sampled from q_e(z|x)
            else:
                return z_fake

        # Generation Mode
        elif x is None and z is not None:
            if 'scm' in self.prior_dist:
                label_z = self.prior(z[:, :self.num_label])  # z after causal layer
                other_z = z[:, self.num_label:]
                z = torch.cat([label_z, other_z], dim=1)
            return self.decoder(z)


class BigJointDiscriminator(nn.Module):
    r'''Big joint discriminator based on SAGAN

    Args:
        latent_dim: latent dimension
        conv_dim: base number of channels
        image_size: image resolution
        fc_size: number of nodes in each fc layers
    '''
    def __init__(self, latent_dim=64, conv_dim=32, image_size=64, fc_size=1024):
        super().__init__()
        self.discriminator = Discriminator(conv_dim, image_size, in_channels=3, out_feature=True)
        self.discriminator_z = Discriminator_MLP(latent_dim, fc_size)
        self.discriminator_j = Discriminator_MLP(conv_dim * 16 + fc_size, fc_size)

    def forward(self, x, z):
        sx, feature_x = self.discriminator(x)
        sz, feature_z = self.discriminator_z(z)
        sxz, _ = self.discriminator_j(torch.cat((feature_x, feature_z), dim=1))
        return (sx + sz + sxz) / 3


def reparameterize(mu, sigma, std=1):
    assert mu.shape == sigma.shape
    eps = mu.new(mu.shape).normal_(0, std)
    return mu + sigma * eps

def kl_div(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()

def gaussian_nll(x_mu, x_logvar, x):
    '''NLL'''
    sigma_inv = (- x_logvar / 2).exp()
    return 0.5 * (x_logvar + ((x - x_mu) * sigma_inv).pow(2) + np.log(2*np.pi)).sum()

def kaiming_init(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        # kaiming_uniform_(m.weight)
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) == nn.BatchNorm1d or type(m) == nn.BatchNorm2d:
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

