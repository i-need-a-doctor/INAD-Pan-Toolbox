import torch, pathlib, scipy.io, os, einops, numpy, mmengine
from typing import Union, List
from .waldprotocal import WaldProtocal


@mmengine.DATASETS.register_module()
class SeperateDataset(torch.utils.data.Dataset):
    def __init__(self, path: Union[str, pathlib.Path], enable_sets: List[str]):
        if isinstance(path, str):
            path = pathlib.Path(path)
        assert path.is_dir()
        self.data_path = []
        for set in enable_sets:
            set_path = path / set
            assert set_path.is_dir()
            ms_path = set_path / "MS_256"
            pan_path = set_path / "PAN_1024"
            assert pan_path.is_dir() and ms_path.is_dir()
            for ms_file in ms_path.iterdir():
                pan_file = pan_path / (ms_file.stem + ".mat")
                assert pan_file.is_file()
                ms_file = ms_path / ms_file
                self.data_path.append((pan_file, ms_file, set))

    def __getitem__(self, index, mer_hrms=False):
        pan_file, ms_file, set = self.data_path[index]
        pan = scipy.io.loadmat(os.fspath(pan_file))
        hhr_pan = pan[list(pan.keys())[-1]]
        ms = scipy.io.loadmat(os.fspath(ms_file))
        hr_ms = ms[list(ms.keys())[-1]]
        hr_pan, lr_ms = WaldProtocal(hhr_pan, hr_ms)
        if mer_hrms:
            hr_pan = hr_ms.mean(axis=-1)
        lr_ms = einops.rearrange(lr_ms, "w h c -> c h w").astype(numpy.float32)
        hr_ms = einops.rearrange(hr_ms, "w h c -> c h w").astype(numpy.float32)
        hr_pan = einops.rearrange(hr_pan, "w h -> 1 h w").astype(numpy.float32)
        return {"gt_hrms": hr_ms, "x_hrpan": hr_pan, "x_lrms": lr_ms}

    def __len__(self):
        return len(self.data_path)


@mmengine.DATASETS.register_module()
class SeperateDataset_4Bands(torch.utils.data.Dataset):
    def __init__(self, path: Union[str, pathlib.Path], enable_sets: List[str]):
        if isinstance(path, str):
            path = pathlib.Path(path)
        assert path.is_dir()
        self.data_path = []
        for set in enable_sets:
            set_path = path / set
            assert set_path.is_dir()
            ms_path = set_path / "MS_256"
            pan_path = set_path / "PAN_1024"
            assert pan_path.is_dir() and ms_path.is_dir()
            for ms_file in ms_path.iterdir():
                pan_file = pan_path / (ms_file.stem + ".mat")
                assert pan_file.is_file()
                ms_file = ms_path / ms_file
                self.data_path.append((pan_file, ms_file, set))

    def __getitem__(self, index, mer_hrms=False):
        pan_file, ms_file, set = self.data_path[index]
        pan = scipy.io.loadmat(os.fspath(pan_file))
        hhr_pan = pan[list(pan.keys())[-1]]
        ms = scipy.io.loadmat(os.fspath(ms_file))
        hr_ms = ms[list(ms.keys())[-1]]
        if "WorldView" in set:
            hr_ms = hr_ms[..., [0, 2, 4, 7]]
        hr_pan, lr_ms = WaldProtocal(hhr_pan, hr_ms)
        if mer_hrms:
            hr_pan = hr_ms.mean(axis=-1)
        lr_ms = einops.rearrange(lr_ms, "w h c -> c h w").astype(numpy.float32)
        hr_ms = einops.rearrange(hr_ms, "w h c -> c h w").astype(numpy.float32)
        hr_pan = einops.rearrange(hr_pan, "w h -> 1 h w").astype(numpy.float32)
        return {"gt_hrms": hr_ms, "x_hrpan": hr_pan, "x_lrms": lr_ms}

    def __len__(self):
        return len(self.data_path)
