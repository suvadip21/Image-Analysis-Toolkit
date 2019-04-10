import numpy as np
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from scipy.misc import comb

from misc.helpers import StdIO as IO
from segmentation.levelset import levelset_helper as H
from filters.spatial import Filter
from misc.helpers import Interactive as IT


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
    Curve evolution as per C_t = (F) n
    """
    def curve_evolve(self, mu=0.1, F=1., color='r', disp_interval=20):
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
            grad_phi = np.gradient(phi)
            grad_phi_mag = np.sqrt(grad_phi[0]**2 + grad_phi[1]**2)

            kappa = H.curvature_central(phi)

            dphi_dt = (F + mu * kappa) * grad_phi_mag          # Gradient decent
            # dt = 0.8/(np.max(np.abs(dphi_dt)) + eps)                                # CFL criteria
            dt = 0.4
            # dphi_dt = F
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
    Active contour without edges - Chan & Vese, TIP'02
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
    Geodesic Active Contours - Casellas et al., IJCV'99
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
        g = 1./(0.01 + grad_sigma_u**(2.0))       # edge feature map
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
                ax.imshow(g, cmap='gray')
                ax.contour(phi, levels=[0], colors=color)
                ax.contour(phi0, levels=[0], colors='g')
                ax.set_axis_off()
                plt.draw()
                plt.pause(1e-5)

        return phi, its

    """
    Legendre Level Set (L2S) - Mukherjee & Acton, SPL'2015
    """
    def l2s(self, k=1, mu=0.2, color='b', disp_interval=20):
        """
        :param k: Order of polynomial
        :param mu: Smoothness parameter
        :param color: contour color
        :param disp_interval: interval to display contour
        :return: levelset, # iterations
        """

        def legendre_basis(img, k):
            '''
            Compute the set of 2D legendre basis functions via outer product of 1d polynomials
            :param img: image of dimension m x n
            :param k: order of polynomial ( k = 0 is chan-vese)
            :return: matrix B: (Nr*Nc)x(k+1)^2, each column is a Legendre polynomial
            '''
            Nr, Nc = img.shape[0], img.shape[1]
            N = Nr*Nc
            B = np.zeros((N, (k+1)*(k+1)))
            def legendre_1d(N, k):
                """
                Compute 1-d legendre polynomial bases
                :param N: length of signal
                :param k: degree of polynomial
                :return: B: N x (k+1)
                """
                xx = np.linspace(-1, 1, N)      # N points in [-1,1]
                p0 = np.ones(N, dtype='float')
                B = np.zeros((N, k+1), dtype='float')
                B[:, 0] = p0                    # first basis is all ones

                for n in range(1, k+1):         # looping through the orders
                    Pn = 0.
                    for m in range(0, n+1):     # m \in [0, n]
                        Pn += (comb(n, m)**2) * ((xx - 1)**(n - m))*((xx + 1)**m)
                    B[:, n] = Pn/(2**n)
                return B

            B_r = legendre_1d(Nr, k)        # 1-d legendre bases, Nr x (k + 1)
            B_c = legendre_1d(Nc, k)        # 1-d legendre bases, Nc x (k + 1)

            # Now compute the 2-d Legendre bases as outer product of 1-d
            idx = 0
            for ii in range(k+1):
                for jj in range(k+1):
                    B[:, idx] = np.outer(B_r[:, ii], B_c[:, jj]).flatten()
                    idx += 1
            return B


        phi = H.mask2phi(self.init_mask)
        phi0 = np.copy(phi)
        u = self.img
        stop = False
        prev_mask = self.init_mask
        its = 0
        c = 0

        B = legendre_basis(u, k)

        if (disp_interval > 0):
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)

        while (its < self.max_iter and stop==False):
            h_phi = heaviside(phi, 2.0)
            delta_phi = dirac(phi, 2.0)

            inside_mask = h_phi
            outside_mask = 1.0 - h_phi
            kappa = H.curvature_central(phi)

            in_vec = inside_mask.flatten()
            out_vec = outside_mask.flatten()
            u_in_vec, u_out_vec = u.flatten() * in_vec, u.flatten() * out_vec

            A1 = np.transpose(B)
            A2 = A1 * np.repeat(in_vec[np.newaxis, :], A1.shape[0], axis=0)
            B2 = A1 * np.repeat(out_vec[np.newaxis, :], A1.shape[0], axis=0)

            c1_vec = np.linalg.solve(np.matmul(A1, np.transpose(A2)), np.matmul(A1, u_in_vec[:, np.newaxis]))
            c2_vec = np.linalg.solve(np.matmul(A1, np.transpose(B2)), np.matmul(A1, u_out_vec[:, np.newaxis]))
            p1_vec = np.matmul(B, c1_vec)
            p2_vec = np.matmul(B, c2_vec)
            p1, p2 = np.reshape(p1_vec, u.shape), np.reshape(p2_vec, u.shape)

            F = -(u - p1)**2 + (u - p2)**2

            dphi_dt = (F/np.max(np.abs(F) + eps) + mu * kappa) * delta_phi          # Gradient decent
            dt = 0.8/(np.max(np.abs(dphi_dt)) + eps)                                # CFL criteria
            phi += dt * dphi_dt
            phi = H.NeumannBoundCond(phi)
            phi = H.sussman(phi, 0.5)

            # -- Convergence criteria
            new_mask = 1.*(phi >=0)
            c = H.convergence(prev_mask, new_mask, self.tolerance, c)
            if c <= 10:
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
    # img = IO.imread_2d('../../image_3.png')
    # img = IO.imread_2d('../../spine_image000002.tif')
    img = IO.imread_2d('../../data/ameoba_1.png')
    mask = IT(img).draw_circle(rad=0, ctr=[])
    # mask = IT(img).draw_polygons(1)
    mask = 1. * (mask > 0)
    # mask = np.zeros(img.shape)
    # mask[10:100, 30:130] = 1.
    # mask[20:100, 20:100] = 1.
    # seg, its = LevelSetFilter(img, init_mask=mask, max_iters=1000, convg_error=1e-5).chan_vese(mu=0.02, color='y', disp_interval=50)
    # seg, its = LevelSetFilter(img, init_mask=mask, max_iters=1500, convg_error=0.001).gac(mu=1.0, c0=-0.5, sigma=4.0, color='b', disp_interval=50)
    seg, its = LevelSetFilter(img, init_mask=mask, max_iters=1000, convg_error=1e-15).l2s(k=1, mu=0.1, color='c', disp_interval=50)
    # seg, its = LevelSetFilter(img, init_mask=mask, max_iters=1000, convg_error=0.05).curve_evolve(F=0.1, mu=0.1, color='c',disp_interval=5)
    IO.imoverlay(img, seg, title='Final result', linewidth=4)





