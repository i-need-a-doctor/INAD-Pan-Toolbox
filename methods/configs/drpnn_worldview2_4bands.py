import torch.nn as nn, torch.optim as optim

_base_ = ["base/base_worldview2_4bands.py"]
custom_imports = dict(imports=_base_.custom_imports["imports"] + ["methods.models.drpnn"], allow_failed_imports=False)

optim_wrapper = dict(optimizer=dict(type=optim.Adam, lr=0.001))
model = dict(type="DRPNN", ms_ch=_base_.ms_ch, pan_ch=_base_.pan_ch, channel=32, criterion=nn.L1Loss())
