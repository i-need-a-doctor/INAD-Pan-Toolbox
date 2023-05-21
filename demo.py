import mmengine, torch.utils.data as data, einops, torch, cv2, numpy
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.optim.scheduler import MultiStepLR

cfg = Config.fromfile("methods/configs/tfnet_worldview2_8bands.py")

model = mmengine.MODELS.build(cfg.model)
dataset = mmengine.DATASETS.build(cfg.dataset)

model.load_state_dict(torch.load("/home/walterd/INAD-Pan-Toolbox/work_dir/tfnet_worldview2_8bands/epoch_100.pth")["state_dict"])
input = dataset[0]
output, residual = model.get_output(input)
output = output[..., [2, 1, 0]]
cv2.imwrite("output.png", (output / 2048 * 255).astype(numpy.uint8))
