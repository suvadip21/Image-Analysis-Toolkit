import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import cv2
from matplotlib import pyplot as plt
import glob
from misc.helpers import StdIP as IP
from misc.helpers import StdIO as IO

class Thresholding:
    def __init__(self, img):
        self.img = IP.im2double(img)

    def otsu_threshold(self):
        img_cv = IP.numpy_to_opencv(self.img)
        tval, th_cv = cv2.threshold(img_cv, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_img = IP.opencv_to_numpy(th_cv)
        return bin_img, tval/255.

    def adaptive_threshold(self, type='gaussian', win_sz=11):
        if type=='gaussian':
            th_option = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        elif type=='mean':
            th_option = cv2.ADAPTIVE_THRESH_MEAN_C
        else:
            th_option = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        img_cv = IP.numpy_to_opencv(self.img)
        bw_cv = cv2.adaptiveThreshold(img_cv, 255, th_option, cv2.THRESH_BINARY, win_sz, 2)
        return IP.opencv_to_numpy(bw_cv)

    def percentile_threshold(self, p1=0., p2=100.):
        """
        Keep values between p1-th and p2-th percentile
        :param p1: lower percentile
        :param p2: higher percentile
        :return: binary image
        """
        low = min(p1, p2)
        high =max(p1, p2)
        low_val = np.percentile(self.img.flatten(), low)
        high_val = np.percentile(self.img.flatten(), high)
        bin_img = np.ones(self.img.shape, dtype='float')
        bin_img[self.img < low_val] = 0.
        bin_img[self.img > high_val] = 0.
        return bin_img




if __name__ == '__main__':
    img = IO.imread_2d('../image_2.png')
    # bin_img, thold = Thresholding(img).otsu_threshold()
    # bin_img = Thresholding(img).adaptive_threshold(type='gaussian', win_sz=55)
    bin_img = Thresholding(img).percentile_threshold(p1=85, p2=100)
    # IO.imshow(bin_img)
    IO.imoverlay(img, bin_img)
    # print thold
    # pts, pt_img = Interactive(img).draw_points(n_pts=- 4)
    # circ = Interactive(img).draw_circle(rad=20)
    # circ = Interactive(img).draw_multi_circle(rad=10)
    # circ = Interactive(img).draw_polygons(n_poly=5)
    # StdIO.imoverlay(img, circ)
    # StdIO.imshow(pt_img)
    print "done"
