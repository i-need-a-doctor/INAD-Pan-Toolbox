import mmengine, torch.utils.data as data
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.optim.scheduler import MultiStepLR

methods = ["drpnn","msdcnn","pannet","tfnet"]
sets = ["worldview2_4bands","worldview2_8bands"]

configs = {}
for set in sets:
    configs[set] = {}
    for method in methods:
        configs[set][method] = Config.fromfile("methods/configs/{}_{}.py".format(method,set))

for set in sets:
    for method in methods:
        cfg = configs[set][method]
        model = mmengine.MODELS.build(cfg.model)
        dataset = mmengine.DATASETS.build(cfg.dataset)
        train_dataloader = data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
        default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=20))
        runner = Runner(
            model=model,
            work_dir="./work_dir/{}_{}".format(method,set),
            train_dataloader=train_dataloader,
            optim_wrapper=cfg.optim_wrapper,
            train_cfg=dict(by_epoch=True, max_epochs=100, val_interval=0),
            default_hooks=default_hooks,
            randomness=dict(seed=42),
        )
        runner.train()
