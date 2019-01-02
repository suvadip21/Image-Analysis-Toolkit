import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import cv2
from matplotlib import pyplot as plt
import glob


eps = 1e-8

class StdIO:
    @staticmethod
    def imread_2d(fname, grayscale=True):
        img = ndimage.imread(fname, flatten=grayscale, mode=None)
        if img.max() > 0:
            img = img/img.max()
        return 1. * img

    @staticmethod
    def imshow(img, title='display', col_map='gray', imadjust=True, pts=[], pt_col='y'):
        fig = plt.figure(title)
        if imadjust:
            img = (img - np.min(img))/(eps + np.max(img) - np.min(img))
        plt.imshow(img, cmap=col_map)
        if len(pts) > 0:
            x, y = list(pts[:, 0]), list(pts[:, 1])
            plt.scatter(x, y, c=pt_col, s=20, zorder=2)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def imoverlay(img, seg, title='Segmentation', col='y', levels=[0], linewidth=2., imadjust=True):
        fig = plt.figure(title)
        if imadjust:
            img = (img - np.min(img))/(eps + np.max(img) - np.min(img))
        plt.imshow(img, cmap='gray')
        plt.contour(seg, levels=levels, colors=col, title=title, linewidth=linewidth)
        plt.tight_layout()
        plt.draw()
        plt.show()

    @staticmethod
    def imread_multiple(folder_path, ext='.png', return_image=True):
        """
        returns the stack of image as stack[ii, :, :]
        shape: (N, Nr, Nc)
        """
        all_data = glob.glob(folder_path + '*' + ext)
        test_img = StdIO.imread_2d(all_data[0])
        Nr, Nc = test_img.shape[0], test_img.shape[1]
        N = len(all_data)
        if return_image:
            stack = np.zeros(N, Nr, Nc)
            for ii in range(N):
                img = StdIO.imread_2d(all_data[ii])
                stack[ii] = img/(eps + np.max(img))
            return stack
        else:
            return all_data


class StdIP:
    @staticmethod
    def imresize(img, orig_res_mm=1., des_res_mm=0.5, interpolation='cubic'):
        frac = orig_res_mm/des_res_mm
        a = sp.misc.imresize(img, frac, interp=interpolation, mode='F')*1.
        if a.max() is not 0:
            a = a/a.max()
        return a

    @staticmethod
    def linstretch(img):
        if img.max() > 0:
            img = (img - np.min(img)) / (eps + np.max(img) - np.min(img))
        return img

    @staticmethod
    def numpy_to_opencv(img):
        img = img/(eps + np.max(img))
        ocv_img = (img * 255).astype('uint8')
        return ocv_img

    @staticmethod
    def opencv_to_numpy(ocv_img):
        np_img = ocv_img.astype('float')/255.
        return np_img


if __name__ == '__main__':
    img = StdIO.imread_2d('../image_1.png')
    # hr_img = Misc.imresize(img, des_res_mm=2.)
    # StdIO.imshow(hr_img)
    # StdIO.imoverlay(img, 1.*(img>0.7), col='r')
