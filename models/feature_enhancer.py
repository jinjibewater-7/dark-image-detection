import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ScaleAwareGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_c = x * self.channel_gate(x)
        avg_map = torch.mean(x_c, dim=1, keepdim=True)
        max_map, _ = torch.max(x_c, dim=1, keepdim=True)
        x_s = x_c * self.spatial_gate(torch.cat([avg_map, max_map], dim=1))
        return x_s


class FeatureEnhancer(nn.Module):
    """
    A lightweight feature enhancement module inspired by FeatEnHancer.
    It keeps the design practical for YOLOv7:
    1) intra-scale enhancement with local convolutions
    2) scale-aware channel/spatial gating
    3) residual refinement to preserve detector-friendly features
    """
    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 2, 64)

        self.pre = ConvBNAct(channels, channels, 3, 1, 1)

        self.branch_local = nn.Sequential(
            ConvBNAct(channels, mid, 3, 1, 1),
            ConvBNAct(mid, channels, 3, 1, 1),
        )

        self.branch_context = nn.Sequential(
            ConvBNAct(channels, mid, 1, 1, 0),
            ConvBNAct(mid, mid, 3, 1, 1),
            nn.Conv2d(mid, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.gate = ScaleAwareGate(channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        identity = x

        x = self.pre(x)
        local_feat = self.branch_local(x)
        context_feat = self.branch_context(F.avg_pool2d(x, kernel_size=2, stride=2))
        context_feat = F.interpolate(context_feat, size=local_feat.shape[-2:], mode='bilinear', align_corners=False)

        fused = local_feat + context_feat
        fused = self.gate(fused)
        fused = self.fuse(fused)

        return identity + fused
