import numpy as np
import scipy.ndimage as nd
# import matplotlib.pyplot as plt
from misc.helpers import StdIO as IO
from matplotlib import pyplot as plt

import levelset_helper as H

eps = np.finfo(float).eps

def heaviside(y, e):
    return 0.5 * (1. + (2. / np.pi) * np.arctan(y / e))

def dirac(x, e):
    return (e/np.pi)/(e**2+ x**2)


class LevelSetFilter:
    def __init__(self, img, init_mask, max_iters=100, convg_error=1e-3):
        self.img = img
        self.init_mask = init_mask
        self.max_iter = max_iters
        self.tolerance = convg_error

    def chan_vese(self, mu=0.1, color='r', disp_interval=20):
        phi = H.mask2phi(self.init_mask)
        phi0 = np.copy(phi)
        u = self.img
        stop = False
        prev_mask = self.init_mask
        its = 0
        c = 0

        if (disp_interval > 0):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        while (its < self.max_iter and stop==False):
            h_phi = heaviside(phi, 2.0)
            delta_phi = dirac(phi, 2.0)

            inside_mask = h_phi
            outside_mask = 1.0 - h_phi
            kappa = H.curvature_central(phi)
            p1 = np.sum(u * inside_mask)/(eps + np.sum(inside_mask))
            p2 = np.sum(u * outside_mask)/(eps + np.sum(outside_mask))
            F = -(u - p1)**2 + (u - p2)**2

            dphi_dt = (F/np.max(np.abs(F) + eps) + mu * kappa) * delta_phi          # Gradient decent
            dt = 0.8/(np.max(np.abs(dphi_dt)) + eps)                                # CFL criteria
            phi += dt * dphi_dt
            phi = H.NeumannBoundCond(phi)
            phi = H.sussman(phi, 0.5)

            new_mask = 1.*(phi >=0)
            c = H.convergence(prev_mask, new_mask, self.tolerance, c)
            if c <= 5:
                its = its + 1
                prev_mask = new_mask
            else:
                stop = True

            if (disp_interval > 0 and np.mod(its, disp_interval)==0):
                ax.cla()
                ax.imshow(u, cmap='gray')
                ax.contour(phi, levels=[0], colors=color)
                ax.contour(phi0, levels=[0], colors='g')
                plt.draw()
                plt.pause(1e-5)

        return phi, its

if __name__=='__main__':
    from misc.helpers import StdIO as IO
    img = IO.imread_2d('../../image_2.png')
    mask = np.zeros(img.shape)
    mask[20:100, 40:120] = 1.
    seg, its = LevelSetFilter(img, init_mask=mask, max_iters=1000, convg_error=0.5).chan_vese(mu=1.0, color='y', disp_interval=50)

    IO.imoverlay(img, seg, title='Final result', linewidth=4)





