###### LIBRARIES #######
import torch
import torch.nn as nn
import torch.nn.functional as F
#-------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.leaky_relu(x + self.block(x), negative_slope=0.2)


class Autoencoder(nn.Module):
    def __init__(self, input_channels=3, latent_channels= 16):
        super().__init__()
        
        # Encoder:Contraction path

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResidualBlock(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            ResidualBlock(512),
            nn.MaxPool2d(2),
        )

        # Spatial latent representation
        self.to_latent = nn.Conv2d(512, latent_channels, kernel_size=1)#----> move to the latent representation
        self.from_latent = nn.Conv2d(latent_channels, 512, kernel_size=1) #----> move from the latent representation

        # Decoder: expansion path

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            ResidualBlock(64),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.Sigmoid()
        )

    # forward pass

    def forward(self, x):
        x = self.encoder(x)  
        z = self.to_latent(x)  
        x = self.from_latent(z)  
        x_recon = self.decoder(x)  
        return x_recon, z