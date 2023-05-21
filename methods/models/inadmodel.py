import torch, einops
from mmengine.model import BaseModel


class INADModel(BaseModel):
    def get_output(self, input):
        x_hrpan = torch.Tensor(einops.rearrange(input["x_hrpan"], "1 h w -> 1 1 h w"))
        x_lrms = torch.Tensor(einops.rearrange(input["x_lrms"], "c h w -> 1 c h w"))
        with torch.no_grad():
            output = self(x_hrpan, x_lrms)
        output = output.squeeze().detach().numpy()
        residual = output - input["gt_hrms"]
        output = einops.rearrange(output, "c h w -> h w c")
        residual = einops.rearrange(residual, "c h w -> h w c")
        return output, residual
