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
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening
from skimage.morphology import erosion, dilation, closing, opening
from skimage.measure import regionprops
from skimage.morphology import disk

class BinaryMorphology:
    """
    Set of methods which will work on binary images only
    """
    def __init__(self, img):
        self.bw = (img > 0.1) * 1.  # force convert to binary

    def bwdilate(self, r=1):
        se = disk(r)
        return binary_dilation(self.bw > 0, selem=se)

    def bwerode(self, r=1):
        se = disk(r)
        return binary_erosion(self.bw > 0, selem=se)

    def bwopen(self, r=1):
        se = disk(r)
        return binary_opening(self.bw > 0, selem=se)

    def bwclose(self, r=1):
        se = disk(r)
        return binary_closing(self.bw > 0, selem=se)

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


class GrayMorphology():
    def __init__(self, img):
        self.img = IP.im2double(img)

    def imdilate(self, r=1):
        se = disk(r)
        return dilation(self.img, selem=se)* 1.

    def imerode(self, r=1):
        se = disk(r)
        return erosion(self.img, selem=se)* 1.

    def imopen(self, r=1):
        se = disk(r)
        return opening(self.img, selem=se)* 1.

    def imclose(self, r=1):
        se = disk(r)
        return closing(self.img, selem=se)* 1.

    def areaopen(self, area=100, l1=0., l2=1.):

        l1 = int(min(255*self.img.min(), 255* (l1/(l1 + 1e-5))))
        l2 = int(max(self.img.max()*255, (255 * (l2 / (l2 + 1e-5)))))
        pixel_range = range(l1, l2 + 1)

        recon_img = np.zeros(self.img.shape)
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1, 2, 1)

        for level in pixel_range:
            tmp_bw_img = 1. * (255. * self.img > level)
            tmp_clean_img = BinaryMorphology(tmp_bw_img).bwareaopen(area=area)
            # ax1.cla()
            # ax1.imshow(tmp_clean_img, cmap='gray')
            # plt.draw()
            # fig.suptitle(str(level))
            # plt.pause(0.001)
            recon_img += tmp_clean_img

        recon_img = IP.linstretch(recon_img/len(pixel_range))
        return recon_img

# TODO: class GrayMorphology: opn-close and gray areaopen
# class Morphology:
#     def __init__(self):



if __name__ == '__main__':
    from segmentation.classic import Thresholding
    img = IO.imread_2d('../image_2.png')
    # bin_img = Thresholding(img).percentile_threshold(p1=75, p2=100)
    # morph_img = BinaryMorphology(bin_img).bwclose(r=4)
    # label_img, n_cc = BinaryMorphology(morph_img).bwlabel()
    #
    # k_largest_img = BinaryMorphology(morph_img).klargestregions(k=1)

    gray_morph_img = GrayMorphology(img).areaopen(area=100, l1=0.1, l2=1.)

    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.imshow(img, cmap='gray')
    # ax.contour(morph_img, levels=[0], colors='r')
    # ax.contour(k_largest_img, levels=range(int(k_largest_img.max())), colors='b')
    # # ax.imshow(label_img)
    # plt.draw()
    # plt.show()

    fig = plt.figure()
    ax1, ax2= fig.add_subplot(1,2,1), fig.add_subplot(1,2,2)
    ax1.imshow(img, cmap='gray')
    ax2.imshow(gray_morph_img, cmap='gray')
    # ax.imshow(label_img)
    plt.draw()
    plt.show()

