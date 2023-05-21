import numpy


def fft(image_torch, scale=8):
    image_torch = numpy.fft.fft2(image_torch)
    image_torch = numpy.fft.fftshift(image_torch)
    image_torch = numpy.log(numpy.abs(image_torch)) * scale
    image_torch[image_torch < 0] = 0
    image_torch[image_torch > 255] = 255
    return image_torch
