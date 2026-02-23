# vae 모델 정의

import torch
import torch.nn as nn

class VAE_MNIST(nn.Module):
    def __init__(self, input_height, input_width, color_channels: int, latent_dim, hidden_dims):
        super(VAE_MNIST, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.color_channels = color_channels
        self.latent_dim = latent_dim
        self.input_dim = self.input_height * self.input_width * self.color_channels

        # Encoder
        encod_layers = []
        in_dim = self.input_dim
        for h in hidden_dims:
            encod_layers.append(nn.Linear(in_dim, h))
            encod_layers.append(nn.ReLU())
            in_dim = h
        encod_layers.append(nn.Linear(in_dim, latent_dim*2))
        self.encoder = nn.Sequential(*encod_layers)
        
        # Decoder
        decod_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decod_layers.append(nn.Linear(in_dim, h))
            decod_layers.append(nn.ReLU())
            in_dim = h
        decod_layers.append(nn.Linear(in_dim, self.input_dim))
        decod_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decod_layers)

        # Sampler
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        # encoding
        x = x.view(-1, self.input_dim)
        encoded_output = self.encoder(x)
        mu, logvar = encoded_output.chunk(2, dim = 1)
        # sampling
        z = self.reparameterize(mu, logvar)
        # decoding
        recon_x = self.decoder(z)
        recon_x = recon_x.view(-1, self.color_channels, self.input_height, self.input_width)
        return recon_x, mu, logvar