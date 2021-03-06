import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import cv2
from matplotlib import pyplot as plt
import glob
import SimpleITK as sitk
eps = 1e-8



def bwdist(a):
    """
    Intermediary function. 'a' has only True/False vals,
    so we convert them into 0/1 values - in reverse.
    True is 0, False is 1, distance_transform_edt wants it that way.
    """
    return ndimage.distance_transform_edt(a == 0)

class StdIO:
    @staticmethod
    def numpy_to_sitk2d(np_arr):
        # Input Array is (nrow x ncol). Returns object shape (ncol x nrow)
        return sitk.GetImageFromArray(np_arr)

    @staticmethod
    def sitk_to_numpy2d(sitk_img):
        # Sitk object shape (ncol x nrow). Returns (nrow x ncol) array
        return sitk.GetArrayFromImage(sitk_img)

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
        plt.axis('off')
        plt.show()

    @staticmethod
    def imoverlay(img, seg, title='Segmentation', col='y', levels=[0], linewidth=2., imadjust=True):
        fig = plt.figure(title)
        if imadjust:
            img = (img - np.min(img))/(eps + np.max(img) - np.min(img))
        plt.imshow(img, cmap='gray')
        plt.contour(seg, levels=levels, colors=col)
        plt.tight_layout()
        plt.draw()
        plt.show()
        plt.axis('off')

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
    def im2double(a):
        a = a.astype(np.float)
        a /= np.abs(a).max()
        return a

    @staticmethod
    def mask2phi(init_a):
        phi = bwdist(init_a) - bwdist(1 - init_a) + StdIP.im2double(init_a) - 0.5
        return -phi

    @staticmethod
    def bwdist(a):
        """
        Intermediary function. 'a' has only True/False vals,
        so we convert them into 0/1 values - in reverse.
        True is 0, False is 1, distance_transform_edt wants it that way.
        """
        return ndimage.distance_transform_edt(a == 0)


    @staticmethod
    def percentile_enhance(img, p1=0., p2=100.):
        """
        Keep values between p1-th and p2-th percentile
        :param p1: lower percentile
        :param p2: higher percentile
        :return: binary image
        """
        low = min(p1, p2)
        high = max(p1, p2)
        low_val = np.percentile(img, low)
        high_val = np.percentile(img, high)
        # print low
        # enh_img = np.copy(img)
        enh_img = (img - low_val)/(high_val - low_val)
        enh_img[enh_img < 0.] = 0.
        enh_img[enh_img > 1.] = 1.
        return enh_img

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

    @staticmethod
    def inpolygon(img, x, y):
        Nr, Nc = img.shape[0], img.shape[1]
        img = np.zeros((Nr, Nc), dtype='float')
        for ii in range(len(y)):
            rowpos = max(0, min(Nr-1, y[ii]))
            colpos = max(0, min(Nc-1, x[ii]))
            img[int(rowpos), int(colpos)] = 1.

        img_ocv = StdIP.numpy_to_opencv(img)
        contour = cv2.findContours(img_ocv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        dist_map = np.zeros(img.shape)
        for ii in range(Nr):
            for jj in range(Nc):
                dist_map[ii, jj] = cv2.pointPolygonTest(contour[1][0][:, 0, :], (jj, ii), True)

        return dist_map

class Interactive:
    def __init__(self, img):
        self.img = img

    def draw_points(self, n_pts=1):
        """
        Click points on the image
        :param n_pts: number if points. Not restricted if n < 0. Terminate by closing the figure. Delete a point with right-click
        :return: list of (x,y) coordinates and a binary image where the selected points are '1'
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_axis_off()
        ax.imshow(self.img, cmap='gray')
        pts = plt.ginput(n=n_pts, show_clicks=True, timeout=0)
        bw_img = np.zeros(self.img.shape, dtype='float')
        for idx in range(len(pts)):
            xx, yy = int(pts[idx][0]), int(pts[idx][1])
            bw_img[yy, xx] = 1.

        return pts, bw_img

    def draw_circle(self, rad=10, ctr=[]):
        """
        Draw a circle by clicking a point and specifying radii. If rad < 0, it needs two points to get radii
        :param rad: radii of circle. Set to negative if using additional point to select radii
        :param ctr: centre of the circle (x, y). User input is used if this is []
        :return: a function f such that f > 0 indicates inside the circle.
        """
        if len(ctr) > 0:
            x0, y0 = ctr[0], ctr[1]
            r = rad
        else:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.imshow(self.img, cmap='gray')
            if rad <= 0: # no radii selected
                pts = plt.ginput(n=2, show_clicks=True, timeout=0)
                r = np.sqrt((pts[0][0] - pts[1][0])**2 + (pts[0][1] - pts[1][1])**2)
                ctr = pts[0]
                x0, y0 = ctr[0], ctr[1]
            else:
                ctr = plt.ginput(n=1, show_clicks=True, timeout=0)
                r = rad
                x0, y0 = ctr[0][0], ctr[0][1]

        Nr, Nc = self.img.shape[0], self.img.shape[1]

        [xx, yy] = np.meshgrid(range(Nc), range(Nr))
        f = (xx - x0)**2 + (yy - y0)**2 - r*r
        return -f

    def draw_multi_circle(self, rad=10):
        """
        Draw multiple circles by clicking. Close figure to stop
        :param rad: radii of circle. Set to negative if using additional point to select radii
        :return: a function f such that f > 0 indicates inside the circle.
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_axis_off()
        ax.imshow(self.img, cmap='gray')

        ctr = plt.ginput(n=-1, show_clicks=True, timeout=0)
        bw_img = np.zeros(self.img.shape, dtype='float')
        Nr, Nc = self.img.shape[0], self.img.shape[1]
        [xx, yy] = np.meshgrid(range(Nc), range(Nr))
        for ii in range(len(ctr)):
            x0, y0 = ctr[ii][0], ctr[ii][1]
            f = (xx - x0) ** 2 + (yy - y0) ** 2 - rad * rad
            bw_img += 1.*(f < 0)
        return 1. * (bw_img > 0)

    def draw_polygons(self, n_poly=1):
        """
        Draw polygons, uses the _roipoly function courtsey: https://github.com/jdoepfert/roipoly.py. Close a polygon by right-click
        :param n_poly: number of polygons to draw. This needs to be prefixed.
        :return: binary image with inside polygon as 1
        """
        from _roipoly import roipoly
        mask = np.zeros(self.img.shape, dtype='float')
        for ii in range(n_poly):
            strng = "Enter polygon:" + str(ii+1)
            fig = plt.figure()
            fig.suptitle(strng)
            ax = fig.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.imshow(self.img, cmap='gray')
            ax.contour(mask, levels=[0])
            roi = roipoly(roicolor='c', fig=fig, ax=ax)
            mask += 1.* roi.getMask(self.img)
        return mask



if __name__ == '__main__':
    img = StdIO.imread_2d('../image_1.png')
    # pts, pt_img = Interactive(img).draw_points(n_pts=- 4)
    # circ = Interactive(img).draw_circle(rad=20, ctr=[125., 125.])
    circ = Interactive(img).draw_multi_circle(rad=10)
    # circ = Interactive(img).draw_polygons(n_poly=5)
    StdIO.imoverlay(img, circ)
    # StdIO.imshow(pt_img)
    print "done"


    # hr_img = Misc.imresize(img, des_res_mm=2.)
    # StdIO.imshow(hr_img)
    # StdIO.imoverlay(img, 1.*(img>0.7), col='r')
