import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.fftpack import fftshift, ifftshift

eps = 1e-8

class Kernel:
    def __init__(self, ker_size=(100, 100)):
        rows, cols = ker_size[0], ker_size[1]
        wy, wx = np.ogrid[-0.5:0.5:1./rows, -0.5:0.5:1./cols]
        self.wy = wy
        self.wx = wx

    def gaussian(self, s0=0.1):
        """
        :param s0: spread of the freqn. domain kernel, [0, 0.25] to avoid aliasing
        :return: frequency domain gaussian kernel
        """
        w = np.sqrt(self.wx**2 + self.wy**2)
        numerator = np.exp(-w**2/(2 * s0**2))
        denominator = numerator.sum() + eps
        return numerator/denominator

    def log_gabor(self, w0=0.2, s0=0.2):
        w = np.sqrt(self.wx**2 + self.wy**2)
        numerator = (np.log10(w/w0))**2
        denominator = 2 * (np.log10(s0))**2
        f = np.e**(-(numerator/denominator))
        return f


def fourier_transform(f, input_centered=False, output_centered=True):
    """
    Performs Fourier transform:
    Inputs:
        f: imput 2D image
        input_centered: kernel center is at image center if True, else, center at (0, 0)
        output_centered: center the frequency transform at image center if True
    """
    if input_centered:
        f = fftshift(f)
    F = fft2(f)
    if output_centered:
        F = fftshift(F)
    return F


def inverse_fourier_transform(F, input_centered=True, output_centered=False):
    """
    Performs inverse Fourier transform:
    Inputs:
        F: imput 2D Frequency domain image
        input_centered: kernel center is at image center if True, else, center at (0, 0)
        output_centered: center the  transform to image center if True, useful if output is a kernel
    """
    if input_centered:
        F = fftshift(F)
    f = ifft2(F)
    if output_centered:
        f = fftshift(f)
    return f


if __name__=='__main__':
    from misc.helpers import StdIO as IO
    img = IO.imread_2d('../image_1.png')
    img_ft = fourier_transform(img)
    gauss_f = Kernel(img.shape).gaussian(s0=0.02)
    gauss_s = inverse_fourier_transform(img_ft*gauss_f, output_centered=False)
    # IO.imshow(np.log(1 + np.abs(img_ft)))
    IO.imshow(np.abs(gauss_s))
