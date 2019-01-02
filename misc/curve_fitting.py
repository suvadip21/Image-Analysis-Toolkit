import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
import cv2
from matplotlib import pyplot as plt

def polyfit_to_components(bw_img, degree=1):
    """
    Description: Fits a ployline to the connected components in the binary image
                This code is written based on the snippet for the  Matlab version of the code for my paper
                "Region Based Segmentation in Presence of Intensity Inhomogeneity Using Legendre Polynomials", IEEE SPL, 2014
    :param bw_img: binary image
    :param degree: degree of polynomial
    :return: polynomial coefficients {c0, c1, ..., cn}, y = c0 + c1*x + ...
    """
    bw = (bw_img)> 0.1
    if (np.sum(bw) > 0):
        [row_idx, col_idx] = np.where(bw > 0)
        X = np.sort(col_idx)
        Y = np.zeros(col_idx.shape, dtype='float')                                                                      # Represent data as point cloud {(xi, yi)}
        for ii in range(len(Y)):
            y_for_xi = np.where(bw[:, X[ii]] == 1)                                                                      # All points corresponding to xi
            Y[ii] = np.mean(y_for_xi)                                                                                   # Take the mean, which is the MMSE for yi, given xi

        A = np.zeros((len(Y), degree+1), dtype='float')
        for jj in range(degree+1):
            A[:, jj] = sorted_col ** jj                                                                                     # A[:,k]=x^k,
            B = np.linalg.lstsq(A, Y, rcond=-1)
            coef = B[0]
    else:
        print "No components present"
        coef = np.zeros((degree+1))

    return coeff