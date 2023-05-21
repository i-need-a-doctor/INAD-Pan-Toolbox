# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;
import torch, torch.nn as nn, mmengine


class Resblock(nn.Module):
    def __init__(self, in_channel=32, mid_channel=32, out_channel=32):
        super(Resblock, self).__init__()
        self.conv20 = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv21 = nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


@mmengine.MODELS.register_module()
class FusionNet(nn.Module):
    def __init__(self, pan_ch, criterion, channel=32):
        super(FusionNet, self).__init__()
        self.spectral_num = pan_ch
        self.criterion = criterion
        self.conv1 = nn.Conv2d(in_channels=pan_ch, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=pan_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(self.res1, self.res2, self.res3, self.res4)

    def forward(self, x_hrpan, x_lrms, up_lrms=None, gt_hrms=None, mode="predict"):
        if up_lrms == None:
            up_lrms = nn.functional.interpolate(x_lrms, scale_factor=4, mode="bicubic", align_corners=False)
        pan_concat = x_hrpan.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_concat, up_lrms)  # Bsx8x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64
        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx8x64x64
        output = output + up_lrms
        if mode == "predict":
            return output
        elif mode == "loss":
            return {"loss": self.criterion(output, gt_hrms)}

    def get_output(self, input):
        import einops

        x_hrpan = torch.Tensor(einops.rearrange(input["x_hrpan"], "1 h w -> 1 1 h w"))
        x_lrms = torch.Tensor(einops.rearrange(input["x_lrms"], "c h w -> 1 c h w"))
        with torch.no_grad():
            output = self(x_hrpan, x_lrms)
        output = output.squeeze().detach().numpy()
        residual = output - input["gt_hrms"]

        output = einops.rearrange(output, "c h w -> h w c")
        residual = einops.rearrange(residual, "c h w -> h w c")
        return output, residual
