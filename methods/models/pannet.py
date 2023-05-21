# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
import torch, torch.nn as nn, mmengine
from .inadmodel import INADModel

class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


@mmengine.MODELS.register_module()
class PanNet(INADModel):
    def __init__(self, ms_ch, pan_ch, criterion, channel=32, reg=True):
        super(PanNet, self).__init__()
        self.criterion = criterion
        self.reg = reg
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.deconv = nn.ConvTranspose2d(in_channels=ms_ch, out_channels=ms_ch, kernel_size=8, stride=4, padding=2, bias=True)
        self.conv1 = nn.Conv2d(in_channels=ms_ch + pan_ch, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=ms_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(self.res1, self.res2, self.res3, self.res4)  # method 2: 4 resnet repeated blocks

    def forward(self, x_hrpan, x_lrms, up_lrms=None, gt_hrms=None, mode="predict"):
        if up_lrms == None:
            up_lrms = nn.functional.interpolate(x_lrms, scale_factor=4, mode="bicubic", align_corners=False)
        output_deconv = self.deconv(x_lrms)
        input = torch.cat([output_deconv, x_hrpan], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64
        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx8x64x64
        output = output + up_lrms
        if mode == "predict":
            return output
        elif mode == "loss":
            return {"loss": self.criterion(output, gt_hrms)}
