import mmengine, torch.utils.data as data
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.optim.scheduler import MultiStepLR

cfg = Config.fromfile("methods/configs/tfnet_worldview2_8bands.py")

model = mmengine.MODELS.build(cfg.model)
dataset = mmengine.DATASETS.build(cfg.dataset)
train_dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=20))
runner = Runner(
    model=model,
    work_dir="./work_dir",
    train_dataloader=train_dataloader,
    optim_wrapper=cfg.optim_wrapper,
    train_cfg=dict(by_epoch=True, max_epochs=100, val_interval=0),
    default_hooks=default_hooks,
    randomness=dict(seed=42),
)
runner.train()
