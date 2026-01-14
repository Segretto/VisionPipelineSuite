import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                            padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class ASPPDecoder(nn.Module):
    def __init__(self, num_classes, in_ch=384, proj_ch=576, branch_ch=288, target_size=(640, 640)):
        """
        num_classes: NC
        in_ch: 384 from DINOv3
        proj_ch: top-level projection (here 576)
        branch_ch: per-branch channel after projection (here 288)
        This configuration yields ≈4.39M params for NC=21.
        """
        super().__init__()
        self.target_size = target_size
        # initial expand: 384 -> proj_ch
        self.initial = nn.Sequential(
            nn.Conv2d(in_ch, proj_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(proj_ch),
            nn.ReLU(inplace=True)
        )

        # project for ASPP branches: proj_ch -> branch_ch
        self.project = nn.Sequential(
            nn.Conv2d(proj_ch, branch_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True)
        )

        # 4 parallel 3x3 conv branches (simulating lightweight ASPP)
        self.branches = nn.ModuleList([
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1, bias=True)
            for _ in range(4)
        ])
        self.branches_bn = nn.ModuleList([nn.BatchNorm2d(branch_ch) for _ in range(4)])
        # concat -> project
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 4, proj_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(proj_ch),
            nn.ReLU(inplace=True)
        )

        # one DS block on proj_ch
        self.ds = DepthwiseSeparable(proj_ch, proj_ch)

        # final classifier head
        self.classifier = nn.Conv2d(proj_ch, num_classes, kernel_size=1, bias=True)

    def forward(self, x,):
        # x: (B, 384, 40, 40)
        x = self.initial(x)
        x = self.project(x)
        branches_out = []
        for conv, bn in zip(self.branches, self.branches_bn):
            y = conv(x)
            y = bn(y)
            y = F.relu(y, inplace=True)
            branches_out.append(y)
        x = torch.cat(branches_out, dim=1)  # B, branch_ch*4, h, w
        x = self.fuse(x)                     # B, proj_ch, h, w
        x = self.ds(x)                       # B, proj_ch, h, w
        logits = self.classifier(x)          # B, NC, h, w

        logits = F.interpolate(logits, size=self.target_size, mode='bilinear', align_corners=False)
        return logits
