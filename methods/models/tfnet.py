import torch.nn as nn, torch, mmengine
from .inadmodel import INADModel


@mmengine.MODELS.register_module()
class TFNet(INADModel):
    def __init__(self, pan_ch, ms_ch, criterion):
        super(TFNet, self).__init__()
        self.criterion = criterion
        self.encoder1_pan = nn.Sequential(
            nn.Conv2d(in_channels=pan_ch, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.encoder2_pan = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2), nn.PReLU())
        self.encoder1_lr = nn.Sequential(
            nn.Conv2d(in_channels=ms_ch, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.encoder2_lr = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2), nn.PReLU())
        self.fusion1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.PReLU(), nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.PReLU()
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.PReLU(),
        )
        self.restore1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.PReLU(),
        )
        self.restore2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.PReLU(),
        )
        self.restore3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=ms_ch, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x_hrpan, x_lrms, up_lrms=None, gt_hrms=None, mode="predict"):
        if up_lrms == None:
            up_lrms = nn.functional.interpolate(x_lrms, scale_factor=4, mode="bicubic", align_corners=False)
        encoder1_pan = self.encoder1_pan(x_hrpan)
        encoder1_lr = self.encoder1_lr(up_lrms)

        encoder2_pan = self.encoder2_pan(encoder1_pan)
        encoder2_lr = self.encoder2_lr(encoder1_lr)

        fusion1 = self.fusion1(torch.cat((encoder2_pan, encoder2_lr), dim=1))
        fusion2 = self.fusion2(fusion1)

        restore1 = self.restore1(fusion2)
        restore2 = self.restore2(torch.cat((restore1, fusion1), dim=1))
        restore3 = self.restore3(torch.cat((restore2, encoder1_lr, encoder1_pan), dim=1))
        output = restore3 + up_lrms
        if mode == "predict":
            return output
        elif mode == "loss":
            return {"loss": self.criterion(output, gt_hrms)}
