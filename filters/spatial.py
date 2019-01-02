import numpy as np
import scipy as sp
from skimage.feature import canny
from misc.helpers import StdIP as ip
from scipy import ndimage as ndi
import cv2

class Kernel():
    def __init__(self, ker_size=(10, 10)):
        r, c = ker_size[0], ker_size[1]
        x, y = np.meshgrid(np.linspace(-c/2., c/2., c), np.linspace(-r/2., r/2., r))
        # y, x = np.ogrid[-0.5:0.5:1./rows, -0.5:0.5:1./cols]
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
        img_cv = ip.numpy_to_opencv(img)
        sz = (12*int(sigma_mm/res) + 1, 12*int(sigma_mm/res) + 1)
        sigma = int(sigma_mm/res)
        blurred_img = cv2.GaussianBlur(img_cv, sz, sigmaX=sigma)
        return ip.opencv_to_numpy(blurred_img)

    @staticmethod
    def log_filter(img, sigma_mm=1., res=1.):
        if sigma_mm > 0:
            u = Filter.gaussian_filter(img, sigma_mm, res)
        else:
            u = np.copy(img)
        g_u = np.gradient(u)
        ux, uy = g_u[1], g_u[0]
        uxx = np.gradient(ux)[1]
        uyy = np.gradient(uy)[0]
        laplacian = uxx + uyy
        return laplacian

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
        blurred_img = cv2.filter2D(img_cv, -1, kernel=ker)
        return ip.opencv_to_numpy(blurred_img)





if __name__=='__main__':
    from misc.helpers import StdIO as IO
    img = IO.imread_2d('../image_1.png')
    # filt_img = Filter.mean_filter(img, rad_mm=10, res=0.2)
    # ker = Kernel(img.shape)
    # gauss_ker = Kernel((100, 100)).gaussian(sigma_mm=10, res=0.2)
    # filt_img = Filter.conv_2d(img, gauss_ker)
    # g, gmag = Filter.gradient_filter(img, sigma_mm=4.)
    # log_filter = Filter.log_filter(img, sigma_mm=15.)
    # log_kernel = Kernel((30, 30)).laplacian_of_gaussian(sigma_mm=2., res=0.5)
    IO.imshow(log_kernel)

