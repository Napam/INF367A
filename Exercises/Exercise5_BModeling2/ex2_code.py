import pandas as pd 
import numpy as np 
from scipy.stats import gamma, poisson, sem, bayes_mvs, multivariate_normal
from matplotlib import pyplot as plt 
np.random.seed(69)

def maxima2d(ax, x, z):
    best_idx = np.argmax(z)
    bestx = x[best_idx]
    bestz = z[best_idx]

    ax.scatter(bestx[0], bestx[1], color='w', marker='x',zorder=10, s=5)
    ax.text(bestx[0], bestx[1] + 1, f'({bestx[0]:.2f}, {bestx[1]:.2f})', color='w')

if __name__ == '__main__':
    def task2b(show=True):
        '''Calculate posterier and plot'''
        df = pd.read_csv('exercise5_2.txt', header=None)
        X = df.values

        sigma = np.array([[1, 0.8],[0.8, 0.2]])
        s0 = np.array([[1, 0],[0, 1]])
        m0 = np.array([15,15])

        N = len(X)
        
        precmat = np.linalg.inv(sigma)
        precmat_s = np.linalg.inv(s0)

        Xsum = X.sum(axis=0)

        A = N*precmat + precmat_s
        b = precmat@Xsum + precmat_s@m0

        sigma_n = np.linalg.inv(A)
        mu_n = sigma_n @ b 

        sigma_n = sigma_n + np.eye(sigma_n.shape[0]) * 1e-2

        if show:
            resolution = 2000
            xrange = np.linspace(8,20,resolution)
            yrange = np.linspace(12,22,resolution)

            Xmesh, Ymesh = np.meshgrid(xrange, yrange)

            points = np.array([Xmesh.ravel(), Ymesh.ravel()]).T

            z_pri = multivariate_normal.pdf(points, m0, s0)
            z_post = multivariate_normal.pdf(points, mu_n, sigma_n)            

            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7,3))
            fig.suptitle('Prior vs posterior')
            ax1.set_title('Prior')
            ax1.imshow(z_pri.reshape((resolution,resolution)), extent=[8, 20, 12, 22], origin='lower')
    
            ax2.set_title('Posterior')
            ax2.imshow(z_post.reshape((resolution, resolution)), extent=[8, 20, 12, 22], origin='lower')
            
            plt.setp([ax1, ax2], xlabel='x', ylabel='y')
            
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            maxima2d(ax2, points, z_post)
            maxima2d(ax1, points, z_pri)

            # plt.savefig('images/2c.pdf')
            plt.show()

    # task2b()