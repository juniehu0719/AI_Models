import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import List
from typing import Tuple



class VarianceScheduler:
    def __init__(self, beta_start: int=0.0001, beta_end: int=0.02, num_steps: int=1000, interpolation: str='linear') -> None:
        self.num_steps = num_steps

        # find the beta valuess by linearly interpolating from start beta to end beta
        if interpolation == 'linear':
            # TODO: complete the linear interpolation of betas here
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif interpolation == 'quadratic':
            # TODO: complete the quadratic interpolation of betas here
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2
        else:
            raise Exception('[!] Error: invalid beta interpolation encountered...')
        

        # TODO: add other statistics such alphas alpha_bars and all the other things you might need here
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x:torch.Tensor, time_step:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device


        self.alpha_bars = self.alpha_bars.to(device)

        # TODO: sample a random noise
        noise = torch.randn_like(x, device=device)

        alpha_bar_t = self.alpha_bars[time_step].view(-1, 1, 1, 1).to(device)

        # TODO: construct the noisy sample
        noisy_input = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

        return noisy_input, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
      super().__init__()

      self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # TODO: compute the sinusoidal positional encoding of the time
        device = time.device
        half_dim = self.dim // 2

        embeddings = torch.arange(half_dim, dtype=torch.float32, device=device)
        embeddings = torch.exp(-math.log(10000) * (2 * embeddings / self.dim))

        time = time[:, None].float()
        sinusoidal = time * embeddings[None, :]

        sin_emb = torch.sin(sinusoidal)
        cos_emb = torch.cos(sinusoidal)
        embeddings = torch.cat([sin_emb, cos_emb], dim=-1)

        return embeddings
    

class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.normalize = normalize
        self.activation = nn.SiLU() if activation is None else activation

        # Conv. layers
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)

        # Dynamically get num_groups for GroupNorm
        num_groups = min(32, out_c) if out_c % 32 == 0 else max(1, out_c // 4)

        self.group_norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_c)
        self.group_norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_c)

    def forward(self, x):
        out = self.conv1(x)
        if self.normalize:
            out = self.group_norm1(out)  
        out = self.activation(out)

        out = self.conv2(out)
        if self.normalize:
            out = self.group_norm2(out)  
        out = self.activation(out)

        return out



class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, 
                 down_channels: List = [64, 128, 128, 128], 
                 up_channels: List = [128, 128, 128, 64], 
                 time_emb_dim: int = 128,
                 num_classes: int = 10) -> None:
        super().__init__()

        self.num_classes = num_classes

        # TODO: time embedding layer
        self.time_mlp = SinusoidalPositionEmbeddings(time_emb_dim)

        # TODO: define the embedding layer to compute embeddings for the labels
        if num_classes > 0:
            self.class_emb = nn.Embedding(num_classes, time_emb_dim)
        else:
            self.class_emb = None

        # Downsampling 
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 32, 32), 1, 10),
            MyBlock((10, 32, 32), 10, 10),
            MyBlock((10, 32, 32), 10, 10),
            nn.Dropout(0.1)  
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 16, 16), 10, 20),
            MyBlock((20, 16, 16), 20, 20),
            MyBlock((20, 16, 16), 20, 20),
            nn.Dropout(0.1)  
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 8, 8), 20, 40),
            MyBlock((40, 8, 8), 40, 40),
            MyBlock((40, 8, 8), 40, 40),
            nn.Dropout(0.1)  
        )
        self.down3 = nn.Conv2d(40, 40, 4, 2, 1)

        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 4, 4), 40, 20),
            MyBlock((20, 4, 4), 20, 20),
            MyBlock((20, 4, 4), 20, 40),
            nn.Dropout(0.1)  
        )

        # Upsampling blocks
        self.up1 = nn.ConvTranspose2d(40, 40, 4, 2, 1)
        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 8, 8), 80, 40),
            MyBlock((40, 8, 8), 40, 20),
            MyBlock((20, 8, 8), 20, 20),
            nn.Dropout(0.1)  
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 16, 16), 40, 20),
            MyBlock((20, 16, 16), 20, 10),
            MyBlock((10, 16, 16), 10, 10),
            nn.Dropout(0.1)  
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 32, 32), 20, 10),
            MyBlock((10, 32, 32), 10, 10),
            MyBlock((10, 32, 32), 10, 10, normalize=False),
            nn.Dropout(0.1)  
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: embed time
        t = self.time_mlp(timestep)

        # TODO: handle label embeddings if labels are avaialble
        if self.class_emb is not None and label is not None:
            label_emb = self.class_emb(label)
            t = t + label_emb

        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 32, 32)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 16, 16)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 8, 8)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 4, 4)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 8, 8)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 8, 8)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 16, 16)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 16, 16)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 32, 32)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 10, 32, 32)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )


class VAE(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 height: int=32, 
                 width: int=32, 
                 mid_channels: List=[32, 32, 32], 
                 latent_dim: int=32, 
                 num_classes: int=10) -> None:
        
        super().__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # NOTE: self.mid_size specifies the size of the image [C, H, W] in the bottleneck of the network
        self.mid_size = [mid_channels[-1], height // (2 ** (len(mid_channels)-1)), width // (2 ** (len(mid_channels)-1))]

        # NOTE: You can change the arguments of the VAE as you please, but always define self.latent_dim, self.num_classes, self.mid_size
        
        # TODO: handle the label embedding here
        self.class_emb = ...
        
        # TODO: define the encoder part of your network
        self.encoder = ...
        
        # TODO: define the network/layer for estimating the mean
        self.mean_net = ...
        
        # TODO: define the networklayer for estimating the log variance
        self.logvar_net = ...

        # TODO: define the decoder part of your network
        self.decoder = ...
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: compute the output of the network encoder
        out = ...

        # TODO: estimating mean and logvar
        mean = self.mean_net(out)
        logvar = self.logvar_net(out)
        
        # TODO: computing a sample from the latent distribution
        sample = self.reparameterize(mean, logvar)

        # TODO: decoding the sample
        out = self.decode(sample, label)

        return out, mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: implement the reparameterization trick: sample = noise * std + mean
        sample = ...

        return sample
    
    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: compute the binary cross entropy between the pred (reconstructed image) and the traget (ground truth image)
        loss = F.binary_cross_entropy(pred, target, reduction='sum')

        return loss
       
    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: compute the KL divergence
        kl_div = -.5 * (logvar.flatten(start_dim=1) + 1 - torch.exp(logvar.flatten(start_dim=1)) - mean.flatten(start_dim=1).pow(2)).sum()

        return kl_div

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            # randomly consider some labels
            labels = torch.randint(0, self.num_classes, [num_samples,], device=device)

        # TODO: sample from standard Normal distrubution
        noise = ...

        # TODO: decode the noise based on the given labels
        out = ...

        return out
    
    def decode(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # TODO: use you decoder to decode a given sample and their corresponding labels
        out = ...

        return out


class LDDPM(nn.Module):
    def __init__(self, network: nn.Module, vae: VAE, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.vae = vae
        self.network = network

        # freeze vae
        self.vae.requires_grad_(False)
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: uniformly sample as many timesteps as the batch size
        t = ...

        # TODO: generate the noisy input
        noisy_input, noise = ...

        # TODO: estimate the noise
        estimated_noise = ...

        # compute the loss (either L1 or L2 loss)
        loss = F.mse_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: implement the sample recovery strategy of the DDPM
        sample = ...

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            labels = torch.randint(0, self.vae.num_classes, [num_samples,], device=device)
        
        # TODO: using the diffusion model generate a sample inside the latent space of the vae
        # NOTE: you need to recover the dimensions of the image in the latent space of your VAE
        sample = ...

        sample = self.vae.decode(sample, labels)
        
        return sample


class DDPM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = x.size(0)

        # TODO: uniformly sample as many timesteps as the batch size
        t = torch.randint(0, self.var_scheduler.num_steps, (batch_size,), device=x.device).long()

        # TODO: generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        # TODO: estimate the noise
        estimated_noise = self.network(noisy_input, t, label)

        # TODO: compute the loss (either L1, or L2 loss)
        loss = F.mse_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: implement the sample recovery strategy of the DDPM

        device = noisy_sample.device

        timestep = timestep.to(self.var_scheduler.betas.device)

        beta_t = self.var_scheduler.betas[timestep].view(-1, 1, 1, 1).to(device)
        alpha_t = self.var_scheduler.alphas[timestep].view(-1, 1, 1, 1).to(device)
        alpha_bar_t = self.var_scheduler.alpha_bars[timestep].view(-1, 1, 1, 1).to(device)

        mean = (
            1 / torch.sqrt(alpha_t)
            * (noisy_sample - beta_t / torch.sqrt(1 - alpha_bar_t) * estimated_noise)
        )

       
        if timestep[0] > 0:  # Only add noise if t > 0
            noise = torch.randn_like(noisy_sample, device=device)
            variance = torch.sqrt(beta_t)
            return mean + variance * noise
        else:  
            return mean
    

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):

        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None

        img_size = (num_samples, 1, 32, 32)  
        samples = torch.randn(img_size, device=device) 

        # TODO: apply the iterative sample generation of the DDPM
        for t in reversed(range(self.var_scheduler.num_steps)):
            timestep = torch.full((num_samples,), t, device=device, dtype=torch.long)

            estimated_noise = self.network(samples, timestep, labels)

            samples = self.recover_sample(samples, estimated_noise, timestep)

        return samples


class DDIM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: uniformly sample as many timesteps as the batch size
        t = ...

        # TODO: generate the noisy input
        noisy_input, noise = ...

        # TODO: estimate the noise
        estimated_noise = ...

        # TODO: compute the loss
        loss = F.l1_loss(estimated_noise, noise)

        return loss
    
    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: apply the sample recovery strategy of the DDIM
        sample = ...

        return sample
    
    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None
        # TODO: apply the iterative sample generation of DDIM (similar to DDPM)
        sample = ...

        return sample