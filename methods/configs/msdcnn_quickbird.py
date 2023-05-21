import torch.nn as nn, torch.optim as optim

_base_ = ["base/base_quickbird.py"]
custom_imports = dict(imports=_base_.custom_imports["imports"] + ["methods.models.msdcnn"], allow_failed_imports=False)

optim_wrapper = dict(optimizer=dict(type=optim.Adam, lr=0.001))
model = dict(type="MSDCNN", ms_ch=_base_.ms_ch, pan_ch=_base_.pan_ch, criterion=nn.L1Loss())
