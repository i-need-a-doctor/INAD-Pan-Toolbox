import cv2, einops


def Bicubic(pan, ms):
    ms = einops.rearrange(ms, "c h w -> w h c")
    up_ms = cv2.resize(ms, dsize=[256, 256], fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    return up_ms
