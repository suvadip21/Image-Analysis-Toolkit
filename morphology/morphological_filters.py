import numpy as np
import scipy as sp
from skimage.feature import canny
from misc.helpers import StdIP as ip
from scipy import ndimage as ndi
import cv2
from matplotlib import pyplot as plt
from misc.helpers import StdIP as IP
from misc.helpers import StdIO as IO
from skimage.morphology import black_tophat, skeletonize, convex_hull_image, remove_small_objects, remove_small_holes, label
from skimage.measure import regionprops

class BinaryMorphology:
    def __init__(self, img):
        self.bw = (img > 0.1) * 1.  # force convert to gray

    def bwskel(self):
        """
        Skeletonize a binary image
        :return:
        """
        return 1. * skeletonize(self.bw)

    def bwareaopen(self, area=20):
        """
        Area opening
        :return:
        """
        return remove_small_objects(self.bw > 0., min_size=area) * 1.

    def bwareaclose(self, area=20):
        """
        Area opening
        :return:
        """
        return remove_small_holes(self.bw > 0., min_size=area) * 1.


    def bwlabel(self):
        """
        Label the connected components
        :return: Labelled objects with colors
        """
        label_img = label(self.bw)
        return 1. * label_img, label_img.max()

    def klargestregions(self, k=1):
        label_img = label(self.bw)
        num_cc = label_img.max()
        k = max(1, min(k, num_cc))
        props = regionprops(label_img)
        area_list = []
        for ii in range(num_cc):
            area_list.append(props[ii].area)

        top_k_comp = np.zeros(label_img.shape)
        if k > 1:
            top_k_labels = np.argsort(area_list)[::-1][0:k] + 1
            for jj in range(k):
                top_k_comp[label_img==top_k_labels[jj]] = top_k_labels[jj]
        else:   # simpler problem of finding largest component
            top_label = np.argmax(area_list) + 1
            top_k_comp[label_img==top_label] = 1
        return top_k_comp


if __name__ == '__main__':
    from segmentation.classic import Thresholding
    img = IO.imread_2d('../image_2.png')
    bin_img = Thresholding(img).percentile_threshold(p1=75, p2=100)
    morph_img = BinaryMorphology(bin_img).bwareaopen(area=30)
    label_img, n_cc = BinaryMorphology(morph_img).bwlabel()

    k_largest_img = BinaryMorphology(morph_img).klargestregions(k=7)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(img, cmap='gray')
    ax.contour(morph_img, levels=[0], colors='r')
    ax.contour(k_largest_img, levels=range(int(k_largest_img.max())), colors='b')
    # ax.imshow(label_img)
    plt.draw()
    plt.show()

    print "num_cc = ", n_cc