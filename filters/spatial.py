import numpy as np
import scipy as sp
from skimage.feature import canny
from misc.helpers import StdIP as ip
from scipy import ndimage as ndi
from scipy import signal
import cv2
from scipy.ndimage import gaussian_laplace
from scipy.ndimage import gaussian_gradient_magnitude
# from skimage.filters import laplace
from skimage.filters import gaussian
from matplotlib import pyplot as plt

class Kernel():
    def __init__(self, ker_size=(10, 10)):
        r, c = ker_size[0], ker_size[1]
        x, y = np.meshgrid(np.linspace(-c/2., c/2., c), np.linspace(-r/2., r/2., r))
        self.y = y
        self.x = x

    def gaussian(self, sigma_mm=1., res=1.):
        """
        :param sigma_mm: gaussian scale parameter in mm units
        :param res: image resolution, pixel/mm
        :return: gaussian kernel centered at image center
        """
        sigma_px = sigma_mm/res
        X = self.x**2 + self.y**2
        guassian_kernel = np.exp(-X/(2*sigma_px**2))
        g = guassian_kernel/guassian_kernel.sum()
        return g

    def laplacian_of_gaussian(self, sigma_mm=1., res=1.):
        u = Kernel.gaussian(self, sigma_mm, res)
        g_u = np.gradient(u)
        ux, uy = g_u[1], g_u[0]
        uxx = np.gradient(ux)[1]
        uyy = np.gradient(uy)[0]
        log = uxx + uyy
        return log

    def average(self, rad_mm=1., res=1.):
        sz = 2*(int(rad_mm/res)) + 1
        A = np.ones((sz, sz), dtype='float')
        return A/(sz*sz)


class Filter():
    @staticmethod
    def mean_filter(img, rad_mm=1., res=1.):
        img_cv = ip.numpy_to_opencv(img)
        sz = 2*int(rad_mm/res) + 1
        blur_img = cv2.blur(img_cv, (sz, sz))
        return ip.opencv_to_numpy(blur_img)

    @staticmethod
    def gaussian_filter(img, sigma_mm=1., res=1.):
        sigma = sigma_mm/res
        blurred_img = gaussian(img, sigma)
        return blurred_img

    @staticmethod
    def log_filter(img, sigma_mm=1., res=1.):
        log = gaussian_laplace(img, sigma=sigma_mm/res)
        return log

    @staticmethod
    def laplacian_filter(img):
        g = np.gradient(img)
        g1 = np.gradient(g[0])
        g2 = np.gradient(g[1])
        gxx, gyy = g1[0], g2[1]
        return gxx + gyy

    @staticmethod
    def median_filter(img, rad_mm=1., res=1.):
        img_cv = ip.numpy_to_opencv(img)
        sz = 2*int(rad_mm/res) + 1
        blur_img = cv2.medianBlur(img_cv, sz)
        return ip.opencv_to_numpy(blur_img)

    @staticmethod
    def gradient_filter(img, sigma_mm=1., res=1.):
        """
        :return: g[0]: gradient along Y, g[1]: gradient along X
        """
        if sigma_mm > 0:
            f = Filter.gaussian_filter(img, sigma_mm, res)
        else:
            f = np.copy(img)
        g = np.gradient(f)
        g_mag = np.sqrt(g[0]**2 + g[1]**2)
        return g, g_mag

    @staticmethod
    def conv_2d(img, ker):
        img_cv = ip.numpy_to_opencv(img)
        if ker.min() > 0:
            blurred_img = cv2.filter2D(img_cv, -1, kernel=ker)
            blurred_img =  ip.opencv_to_numpy(blurred_img)
        else:   # I found a open-cv bug when kernel is negative
            blurred_img = cv2.filter2D(img_cv/255., -1, kernel=ker)
        return blurred_img



if __name__=='__main__':
    from misc.helpers import StdIO as IO
    img = IO.imread_2d('/home/suvadip21/Documents/Codes/image_analysis_toolkit/data/ameoba_1.png')

    std = 10.

    ker = Kernel((int(6*std), int(6*std)))
    gauss_ker = ker.gaussian(sigma_mm=std)
    log_ker = ker.laplacian_of_gaussian(sigma_mm=std)
    mean_ker = ker.average(rad_mm=std)
    # smooth_conv = Filter.conv_2d(img, gauss_ker)
    log_conv = Filter.conv_2d(img, log_ker)

    log_img = Filter.log_filter(img, sigma_mm=std)
    # g, gmag = Filter.gradient_filter(smooth_img, sigma_mm=0.)
    # log_img = Filter.log_filter(img, sigma_mm=std)
    # laplacian_img = Filter.laplacian_filter(smooth_img)

    f = plt.figure()
    ax1 = f.add_subplot(1,2,1)
    ax2 = f.add_subplot(1, 2, 2)
    ax1.imshow(log_img, cmap='gray')
    ax1.set_title('using filter')
    ax2.imshow(log_conv, cmap='gray')
    ax2.set_title('using convolution')
    plt.show()