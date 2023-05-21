import pathlib, cv2, einops, numpy, torch, os, pandas, pathlib, seaborn as sns, mmengine
from methods.models import PCA, GSA, MTF_GLP_HPM, Bicubic
from utils.vis.fft import fft
from utils.vis.linestretch import linestretch
from mmengine.config import Config

dataset_name = "worldview2_4bands"
index = 1
output_folder = pathlib.Path("output")
output_folder.mkdir(exist_ok=True)

# Load dataset
cfg = Config.fromfile(f"methods/configs/base/base_{dataset_name}.py")
dataset = mmengine.DATASETS.build(cfg.dataset)

funcs = [("Bicubic", Bicubic), ("PCA", PCA), ("GSA", GSA), ("MTF_GLP_HPM", MTF_GLP_HPM)]
# Load DL models
DL_models = {}
DL_method_names = ["drpnn", "msdcnn", "pannet", "tfnet"]
for DL_method_name in DL_method_names:
    cfg = Config.fromfile(f"methods/configs/{DL_method_name}_{dataset_name}.py")
    model = mmengine.MODELS.build(cfg.model)
    model.load_state_dict(torch.load(f"checkpoints/{DL_method_name}_{dataset_name}.pth")["state_dict"])
    funcs += [(DL_method_name, model.get_output)]

frame = dataset[index]
hr_pan = frame["x_hrpan"]
lr_ms = frame["x_lrms"]
hr_ms = frame["gt_hrms"]

residual_maps = {}
residual_max = 0
residual_df = pandas.DataFrame({"index": [], "mse_value": [], "channel": [], "func": []})
for name, func in funcs:
    output = func(hr_pan / 2047, lr_ms / 2047) * 2047
    residual_map = numpy.abs(output - hr_ms)
    residual_maps[name] = residual_map
    residual_max = max(residual_max, residual_map.max())

    # 生成折线图源数据
    down_residual = residual_map
    # down_residual = cv2.resize(residual_map, dsize=[8, 8], interpolation=cv2.INTER_NEAREST)
    for channel_num in range(down_residual.shape[-1]):
        channel = down_residual[..., channel_num] * -1
        channel = channel.reshape(-1)[:64]
        channel = pandas.DataFrame(channel)
        channel = channel.rename(columns={0: "mse_value"})
        channel.insert(0, "index", range(0, len(channel)))
        channel["channel"] = f"Band #{channel_num+1}"
        channel["func"] = name
        residual_df = pandas.concat([residual_df, channel], axis=0)

    output = linestretch(output)
    output = (output * 255).astype(numpy.int8)
    cv2.imwrite(os.fspath(output_folder / f"result_{name}.png"), output[..., :3].astype(numpy.uint8))
    fft_res = fft(numpy.mean(output, axis=-1))
    # fft_res = linestretch(fft_res)
    # fft_res = (fft_res*255).astype(numpy.uint8)
    cv2.imwrite(os.fspath(output_folder / f"fft_{name}.png"), fft_res)

for name, residual_map in residual_maps.items():
    residual_map = residual_map / residual_max
    residual_map = residual_map * 255
    residual_map = numpy.mean(residual_map, axis=-1).astype(numpy.uint8)
    residual_map = cv2.applyColorMap(residual_map, cv2.COLORMAP_JET)
    cv2.imwrite(os.fspath(output_folder / f"residual_{name}.png"), residual_map)

sns.set_theme(style="darkgrid")
palette = sns.color_palette("rocket_r")
sns.relplot(
    data=residual_df,
    x="index",
    y="mse_value",
    col="channel",
    hue="func",
    kind="line",
    size_order=["T1", "T2"],
    palette=palette,
    height=5,
    aspect=1,
    errorbar="sd",
    facet_kws=dict(sharex=False),
).set(xticklabels=[]).set_axis_labels("Patch Index", "Intensity of Error").set_titles("{col_name}").savefig(os.fspath(output_folder / "test.png"))
