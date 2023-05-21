import torch.nn as nn, torch.optim as optim

_base_ = ["base/base_worldview2_4bands.py"]
custom_imports = dict(imports=_base_.custom_imports["imports"] + ["methods.models.tfnet"], allow_failed_imports=False)

optim_wrapper = dict(optimizer=dict(type=optim.Adam, lr=1e-4))
model = dict(type="TFNet", pan_ch=_base_.pan_ch, ms_ch=_base_.ms_ch, criterion=nn.L1Loss())
