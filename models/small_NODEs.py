import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint 
from tqdm import tqdm

# 1. Define Neural ODE dynamics 
# class ConvLatentODEFunc(nn.Module):
#     def __init__(self, channels, residual_scale=1.0):
#         super().__init__()
#         self.residual_scale = residual_scale

#         self.norm1 = nn.GroupNorm(num_groups=int(channels/2), num_channels=channels)
#         self.conv1 = nn.Conv2d(channels, 128, kernel_size=3, padding=1)
#         self.act1 = nn.SiLU()
#         self.dropout1 = nn.Dropout2d(0.05)

#         self.norm2 = nn.GroupNorm(num_groups=int(128/2), num_channels=128)
#         self.conv2 = nn.Conv2d(128, channels, kernel_size=3, padding=1)

#         self._init_weights()

#     def forward(self, t, z):
#         residual = z

#         out = self.norm1(z)
#         out = self.act1(self.conv1(out))
#         out = self.dropout1(out)

#         out = self.norm2(out)
#         out = self.conv2(out)

#         return residual + self.residual_scale * out

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)


# class LatentODEModel(nn.Module):
#     def __init__(self, ode_func, method='', rtol=1e-5, atol=1e-5):
#         super().__init__()
#         self.ode_func = ode_func
#         self.method = method
#         self.rtol = rtol
#         self.atol = atol

#     def forward(self, z0, t):
#         return odeint(self.ode_func, z0, t, method=self.method, rtol=self.rtol, atol=self.atol)

class ConvLatentODEFunc(nn.Module):
    def __init__(self, channels, residual_scale=1.0):
        super().__init__()
        self.residual_scale = residual_scale

        self.norm1 = nn.GroupNorm(num_groups=int(channels/2), num_channels=channels)
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.act1 = nn.SiLU()
        self.dropout1 = nn.Dropout2d(0.05)

        self.norm2 = nn.GroupNorm(4, 64)
        self.conv2 = nn.Conv2d(64, channels, kernel_size=3, padding=1)

        self._init_weights()

    def forward(self, t, z):
        residual = z

        out = self.norm1(z)
        out = self.act1(self.conv1(out))
        out = self.dropout1(out)

        out = self.norm2(out)
        out = self.conv2(out)

        return residual + self.residual_scale * out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class LatentODEModel(nn.Module):
    def __init__(self, ode_func, method='dopri5', rtol=1e-5, atol=1e-5):
        super().__init__()
        self.ode_func = ode_func
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, z0, t):
        return odeint(self.ode_func, z0, t, method=self.method, rtol=self.rtol, atol=self.atol)

