import torch, einops, math, numpy


def linestretch(images, tol=None):
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().squeeze().numpy()
        images = einops.rearrange(images, "C H W -> H W C")
    if tol is None:
        tol = [0.01, 0.995]
    if images.ndim == 3:
        h, w, channels = images.shape
    else:
        images = numpy.expand_dims(images, axis=-1)
        h, w, channels = images.shape
    images = images.astype(numpy.float16)
    N = h * w
    for c in range(channels):
        image = numpy.float32(numpy.round(images[:, :, c])).reshape(N, 1)
        image = image - numpy.min(image)
        hb, levelb = numpy.histogram(image, bins=math.ceil(image.max() - image.min()))
        chb = numpy.cumsum(hb, 0)
        levelb_center = levelb[:-1] + (levelb[1] - levelb[0]) / 2
        lbc_min, lbc_max = levelb_center[chb > N * tol[0]][0], levelb_center[chb < N * tol[1]][-1]
        image = numpy.clip(image, a_min=lbc_min, a_max=lbc_max)
        image = (image - lbc_min) / (lbc_max - lbc_min)
        images[..., c] = numpy.reshape(image, (h, w))
    images = numpy.squeeze(images)
    return images
