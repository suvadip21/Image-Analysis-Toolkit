import numpy as np
import scipy.ndimage as nd
from misc.helpers import StdIO as IO
from matplotlib import pyplot as plt
import levelset_helper as H
from filters.spatial import Filter

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

    """
    Active contour without edges - Chan & Vese
    """
    def chan_vese(self, mu=0.1, color='r', disp_interval=20):
        """
        :param mu: Smoothness parameteer
        :param color: contour color
        :param disp_interval: contour display interval
        :return: levelset, # iterations
        """
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

            # -- Convergence criteria
            new_mask = 1.*(phi >=0)
            c = H.convergence(prev_mask, new_mask, self.tolerance, c)
            if c <= 5:
                its = its + 1
                prev_mask = new_mask
            else:
                stop = True

            # -- Display of curve
            if (disp_interval > 0 and np.mod(its, disp_interval)==0):
                ax.cla()
                ax.imshow(u, cmap='gray')
                ax.contour(phi, levels=[0], colors=color)
                ax.contour(phi0, levels=[0], colors='g')
                plt.draw()
                plt.pause(1e-5)

        return phi, its

    """
    Geodesic Active Contours - Sapiro et al.
    """
    def gac(self, mu=1., c0=-0.8, sigma=2., color='b', disp_interval=20):
        """
        :param mu: mu=0 implies mean curvature motion
        :param c0: advection term
        :param sigma: std. deviation of gaussian for the edge feature map
        :param color: contour display color
        :param disp_interval: contour diaplay interval. Not shown if negative
        :return: levelset, # iterations
        """
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

        # -- Compute the image derived feature map
        _, grad_sigma_u = Filter.gradient_filter(u, sigma_mm=sigma)
        g = 1./(0.01 + grad_sigma_u**(1.0))       # edge feature map
        g = (g - g.min())/(g.max() - g.min())
        grad_g = np.gradient(g)

        while (its < self.max_iter and stop==False):
            grad_phi = np.gradient(phi)
            grad_phi_mag = np.sqrt(grad_phi[0]**2 + grad_phi[1]**2)
            kappa = H.curvature_central(phi)

            dphi_dt1 = g * grad_phi_mag                                     # Advection term
            dphi_dt2 = g * grad_phi_mag * kappa;                            # Curvature term
            dphi_dt3 = grad_g[0] * grad_phi[0]+ grad_g[1] * grad_phi[1]     # Edge attraction term

            # dphi_dt1 = dphi_dt1 / (eps + np.max(np.abs(dphi_dt1)))
            # dphi_dt2 = dphi_dt2 / (eps + np.max(np.abs(dphi_dt2)))
            # dphi_dt3 = dphi_dt3 / (eps + np.max(np.abs(dphi_dt3)))

            F = c0 * dphi_dt1 + mu * dphi_dt2 + dphi_dt3
            dt = 0.6/(np.max(np.abs(F)) + eps)                              # CFL criteria

            phi += dt * F
            phi = H.NeumannBoundCond(phi)
            phi = H.sussman(phi, 0.5)

            #-- Convergence criteria
            new_mask = 1.*(phi >=0)
            c = H.convergence(prev_mask, new_mask, self.tolerance, c)
            if c <= 5:
                its = its + 1
                prev_mask = new_mask
            else:
                stop = True

            # -- Display of curve
            if (disp_interval > 0 and np.mod(its, disp_interval)==0):
                ax.cla()
                ax.imshow(u, cmap='gray')
                ax.contour(phi, levels=[0], colors=color)
                ax.contour(phi0, levels=[0], colors='g')
                ax.set_axis_off()
                plt.draw()
                plt.pause(1e-5)

        return phi, its

if __name__=='__main__':
    from misc.helpers import StdIO as IO
    img = IO.imread_2d('../../image_2.png')
    mask = np.zeros(img.shape)
    mask[40:80, 40:100] = 1.
    # seg, its = LevelSetFilter(img, init_mask=mask, max_iters=1000, convg_error=0.5).chan_vese(mu=1.0, color='y', disp_interval=50)
    seg, its = LevelSetFilter(img, init_mask=mask, max_iters=1500, convg_error=0.1).gac(mu=1.0, c0=-1.5, sigma=2.0, color='b', disp_interval=50)
    IO.imoverlay(img, seg, title='Final result', linewidth=4)





