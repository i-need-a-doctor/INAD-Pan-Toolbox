import torch, einops
from mmengine.model import BaseModel


class INADModel(BaseModel):
    def get_output(self, pan, hs):
        x_hrpan = torch.Tensor(einops.rearrange(pan, "1 w h -> 1 1 h w"))
        x_lrms = torch.Tensor(einops.rearrange(hs, "c w h -> 1 c h w"))
        with torch.no_grad():
            output = self(x_hrpan, x_lrms)
        output = output.squeeze().detach().numpy()
        output = einops.rearrange(output, "c h w -> w h c")
        return output
