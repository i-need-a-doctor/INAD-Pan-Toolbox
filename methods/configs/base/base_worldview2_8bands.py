_base_ = ["base_8bands.py"]
custom_imports = dict(imports=["datasets.seperatedataset"], allow_failed_imports=False)

dataset = dict(
    type="SeperateDataset",
    path=_base_.datasets_root,
    enable_sets=["WorldView-2"],
)