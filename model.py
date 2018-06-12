import torch
import torch.nn as nn
import gc
import math

n_cls = 22

class Seg(nn.Module):
    def __init__(self):
        super(Seg, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(2048, 32, kernel_size=1),
            nn.Conv2d(1024, 32, kernel_size=1),
            nn.Conv2d(512, 16, kernel_size=1),
            nn.Conv2d(256, 8, kernel_size=1),
            nn.Conv2d(64, 4, kernel_size=1)]
        )
        self.puts = nn.ModuleList(
            [nn.Conv2d(64, 64, kernel_size=1),
            nn.Sequential(
                nn.Conv2d(64+16,64+16, kernel_size=1),
                nn.UpsamplingBilinear2d(scale_factor=2)
            ),
            nn.Sequential(
                nn.Conv2d(64+16+8, 64+16+8, kernel_size=1),
                nn.UpsamplingBilinear2d(scale_factor=2)
            ),
            nn.Sequential(
                nn.Conv2d(64+16+8+4, n_cls, kernel_size=1),
                nn.UpsamplingBilinear2d(scale_factor=2)
            )]
        )
        self.upscale4 = nn.Conv2d(2048, 32, kernel_size=1)
        self.upscale3 = nn.Conv2d(1024, 32, kernel_size=1)
        self.upscale2 = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=2))
        self.upscale1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=2))
        self.upscale0 = nn.Sequential(
            nn.Conv2d(128, 21, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.normal_(0)

    def forward(self, feats):
        feats = feats[::-1]
        for i in range(5):
            feats[i] = self.convs[i](feats[i])
        x = feats[0]
        i = 1
        for l in self.puts:
            x = torch.cat((x, feats[i]), 1)
            x = l(x)
            i += 1
        return x