import torch.nn as nn, torch.optim as optim

_base_ = ["base/base_quickbird.py"]
custom_imports = dict(imports=_base_.custom_imports["imports"] + ["methods.models.pannet"], allow_failed_imports=False)

optim_wrapper = dict(optimizer=dict(type=optim.Adam, lr=0.001))
model = dict(type="PanNet", spectral_num=_base_.ms_ch, criterion=nn.MSELoss())
