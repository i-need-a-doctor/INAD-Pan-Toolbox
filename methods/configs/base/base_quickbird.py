_base_ = ["base_4bands.py"]
custom_imports = dict(imports=["datasets.seperatedataset"], allow_failed_imports=False)

dataset = dict(
    type="SeperateDataset_4Bands",
    path=_base_.datasets_root,
    enable_sets=["QuickBird"],
)