# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, Ran Ran, LiangJian Deng
# @reference:
import torch, torch.nn as nn, mmengine
from mmengine.model import BaseModel
from .inadmodel import INADModel

class Repeatblock(nn.Module):
    def __init__(self):
        super(Repeatblock, self).__init__()

        channel = 32  # input_channel =
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=7, stride=1, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs = self.relu(self.conv2(x))

        return rs


@mmengine.MODELS.register_module()
class DRPNN(INADModel):
    def __init__(self, ms_ch,pan_ch, channel, criterion):
        super(DRPNN, self).__init__()
        self.criterion = criterion
        self.conv1 = nn.Conv2d(in_channels=ms_ch+ pan_ch, out_channels=channel, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=ms_ch + pan_ch, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv3 = nn.Conv2d(in_channels=ms_ch + pan_ch, out_channels=ms_ch, kernel_size=7, stride=1, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
            Repeatblock(),
        )

    def forward(self, x_hrpan, x_lrms, up_lrms=None, gt_hrms=None, mode="predict"):  # x= lms; y = pan
        if up_lrms == None:
            up_lrms = nn.functional.interpolate(x_lrms, scale_factor=4, mode="bicubic", align_corners=False)
        input = torch.cat([up_lrms, x_hrpan], 1)  # Bsx9x64x64
        rs = self.relu(self.conv1(input))  # Bsx64x64x64
        rs = self.backbone(rs)  # backbone!  Bsx64x64x64
        out_res = self.conv2(rs)  # Bsx9x64x64
        output1 = torch.add(input, out_res)  # Bsx9x64x64
        output = self.conv3(output1)  # Bsx8x64x64
        if mode == "predict":
            return output
        elif mode == "loss":
            return {"loss": self.criterion(output, gt_hrms)}