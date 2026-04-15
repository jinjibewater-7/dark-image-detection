import torch
import torch.nn as nn


class LightEAM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.att = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self,x):

        att = self.att(x)

        return x * att + x