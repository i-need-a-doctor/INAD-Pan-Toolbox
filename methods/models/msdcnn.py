# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, Ran Ran, LiangJian Deng
# @reference:
import torch, torch.nn as nn, mmengine
from .inadmodel import INADModel


@mmengine.MODELS.register_module()
class MSDCNN(INADModel):
    def __init__(self, pan_ch, ms_ch, criterion):
        super(MSDCNN, self).__init__()
        self.criterion = criterion
        input_channel = ms_ch + pan_ch
        output_channel = ms_ch
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=60, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv2_1 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2_3 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv3 = nn.Conv2d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_1 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4_3 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=7, stride=1, padding=3, bias=True)
        self.conv5 = nn.Conv2d(in_channels=30, out_channels=output_channel, kernel_size=5, stride=1, padding=2, bias=True)
        self.shallow1 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
        self.shallow2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
        self.shallow3 = nn.Conv2d(in_channels=32, out_channels=output_channel, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x_hrpan, x_lrms, up_lrms=None, gt_hrms=None, mode="predict"):  # x: lms; y: pan
        if up_lrms == None:
            up_lrms = nn.functional.interpolate(x_lrms, scale_factor=4, mode="bicubic", align_corners=False)
        concat = torch.cat([up_lrms, x_hrpan], 1)  # Bsx9x64x64
        out1 = self.relu(self.conv1(concat))  # Bsx60x64x64
        out21 = self.conv2_1(out1)  # Bsx20x64x64
        out22 = self.conv2_2(out1)  # Bsx20x64x64
        out23 = self.conv2_3(out1)  # Bsx20x64x64
        out2 = torch.cat([out21, out22, out23], 1)  # Bsx60x64x64
        out2 = self.relu(torch.add(out2, out1))  # Bsx60x64x64
        out3 = self.relu(self.conv3(out2))  # Bsx30x64x64
        out41 = self.conv4_1(out3)  # Bsx10x64x64
        out42 = self.conv4_2(out3)  # Bsx10x64x64
        out43 = self.conv4_3(out3)  # Bsx10x64x64
        out4 = torch.cat([out41, out42, out43], 1)  # Bsx30x64x64
        out4 = self.relu(torch.add(out4, out3))  # Bsx30x64x64
        out5 = self.conv5(out4)  # Bsx8x64x64
        shallow1 = self.relu(self.shallow1(concat))  # Bsx64x64x64
        shallow2 = self.relu(self.shallow2(shallow1))  # Bsx32x64x64
        shallow3 = self.shallow3(shallow2)  # Bsx8x64x64
        output = torch.add(out5, shallow3)  # Bsx8x64x64
        if mode == "predict":
            return output
        elif mode == "loss":
            return {"loss": self.criterion(output, gt_hrms)}
