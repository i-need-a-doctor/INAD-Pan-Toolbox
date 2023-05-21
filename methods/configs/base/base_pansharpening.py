import pathlib
# Spatial resolution factor of HR to LR
factor = 4
# HR spatial resolution of each patch
hr_patch_size = 256
# Number of channels of PAN
pan_ch = 1
# Datasets root path
datasets_root = pathlib.Path(r"/home/walterd/HyperTransformer/datasets/Satellite_Dataset")
assert datasets_root.is_dir()