import torch
import torch.nn as nn


class LearnableEnhance(nn.Module):
    """
    Detection-oriented low-light enhancement.

    Design goals:
    1. Only mainly enhance dark regions.
    2. Keep bright regions as unchanged as possible.
    3. Use residual fusion to reduce distribution shift for the detector.
    """

    def __init__(self):
        super().__init__()

        # Global per-channel curve parameters
        self.a = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1))
        self.gamma = nn.Parameter(torch.tensor([0.8, 0.8, 0.8]).view(1, 3, 1, 1))
        self.b = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1))

        # Learnable dark-region mask, single-channel for stability
        self.mask_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        # Very light detail branch to avoid over-smoothing after enhancement
        self.detail_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = torch.clamp(x, 1e-6, 1.0)

        # Conservative parameter range to reduce detector disturbance
        a = torch.clamp(self.a, 0.8, 1.5)
        gamma = torch.clamp(self.gamma, 0.6, 1.4)
        b = torch.clamp(self.b, -0.05, 0.05)

        # Curve-based illumination enhancement
        x_curve = a * torch.pow(x, gamma) + b
        x_curve = torch.clamp(x_curve, 0.0, 1.0)

        # Weak detail compensation
        detail = 0.05 * self.detail_conv(x)
        x_detail = torch.clamp(x + detail, 0.0, 1.0)

        # Dark prior: high in dark areas, low in bright areas
        gray = torch.mean(x, dim=1, keepdim=True)
        dark_mask = torch.exp(-4.0 * gray)

        # Learnable refinement of the dark-region mask
        learn_mask = self.mask_conv(x)
        mask = torch.clamp(dark_mask * learn_mask, 0.0, 1.0)

        # Conservative fusion: enhancement dominates, detail is auxiliary
        x_enh = 0.85 * x_curve + 0.15 * x_detail

        # Residual enhancement to keep detector-friendly distribution
        x_out = x + mask * (x_enh - x)
        x_out = torch.clamp(x_out, 0.0, 1.0)
        return x_out
