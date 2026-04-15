import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class FeatureRefine(nn.Module):
    def __init__(self, channels, expand=2, residual_scale=0.3):
        super().__init__()
        hidden = channels * expand
        self.residual_scale = residual_scale

        self.refine = nn.Sequential(
            ConvBNAct(channels, hidden, 3, 1),
            ConvBNAct(hidden, channels, 3, 1),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.refine(x)
        out = x + self.residual_scale * y
        return self.act(out)


class CrossScaleGate(nn.Module):
    def __init__(self, low_c, high_c):
        super().__init__()

        self.high_proj = nn.Sequential(
            nn.Conv2d(high_c, low_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(low_c),
            nn.ReLU(inplace=True)
        )

        self.gate = nn.Sequential(
            nn.Conv2d(low_c * 2, low_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(low_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_c, low_c, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, low_feat, high_feat):
        high_up = F.interpolate(
            high_feat,
            size=low_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        high_up = self.high_proj(high_up)

        alpha = self.gate(torch.cat([low_feat, high_up], dim=1))
        out = low_feat + alpha * high_up
        return out


class FeatEnhancerLite(nn.Module):
    def __init__(self, c3=256, c4=512, c5=1024, residual_scale=0.3):
        super().__init__()

        self.refine3 = FeatureRefine(c3, expand=2, residual_scale=residual_scale)
        self.refine4 = FeatureRefine(c4, expand=2, residual_scale=residual_scale)
        self.refine5 = FeatureRefine(c5, expand=2, residual_scale=residual_scale)

        self.gate54 = CrossScaleGate(c4, c5)
        self.gate43 = CrossScaleGate(c3, c4)

    def forward(self, p3, p4, p5):
        p3_ori = p3
        p4_ori = p4
        p5_ori = p5

        p3 = self.refine3(p3)
        p4 = self.refine4(p4)
        p5 = self.refine5(p5)

        p4 = self.gate54(p4, p5)
        p3 = self.gate43(p3, p4)

        loss3 = torch.mean(torch.abs(p3 - p3_ori))
        loss4 = torch.mean(torch.abs(p4 - p4_ori))
        loss5 = torch.mean(torch.abs(p5 - p5_ori))
        feat_loss = loss3 + loss4 + loss5

        return p3, p4, p5, feat_loss