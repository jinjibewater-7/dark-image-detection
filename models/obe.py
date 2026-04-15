import torch
import torch.nn as nn


class LightOBE(nn.Module):
    """
    Lightweight OBE-style front-end for low-light detection.
    Keeps input/output in [0, 1].
    """
    def __init__(self, channels=16):
        super().__init__()
        self.wb = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1))
        self.gamma = nn.Parameter(torch.tensor([0.8, 0.8, 0.8]).view(1, 3, 1, 1))
        self.bias = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1))

        self.mask_net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 1),
            nn.Sigmoid()
        )

        self.print_count = 0
        self.print_interval = 200

    def forward(self, x):
        x = torch.clamp(x, 1e-6, 1.0)

        wb = torch.clamp(self.wb, 0.7, 1.5)
        gamma = torch.clamp(self.gamma, 0.4, 1.5)
        bias = torch.clamp(self.bias, -0.1, 0.1)

        x_wb = torch.clamp(x * wb, 0.0, 1.0)
        x_gamma = torch.clamp(torch.pow(x_wb, gamma), 0.0, 1.0)
        x_enh = torch.clamp(x_gamma + bias, 0.0, 1.0)

        mask = self.mask_net(x)

        if self.training and self.print_count % self.print_interval == 0:
            print("\n===== OBE Parameters =====")
            print("wb     :", self.wb.view(-1).detach().cpu().numpy())
            print("gamma  :", self.gamma.view(-1).detach().cpu().numpy())
            print("bias   :", self.bias.view(-1).detach().cpu().numpy())
            print("mask   :", mask.mean().item())
            print("==========================\n")
        self.print_count += 1

        out = (1.0 - mask) * x + mask * x_enh
        return torch.clamp(out, 0.0, 1.0)
