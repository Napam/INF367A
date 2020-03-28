import numpy as np 
import pandas as pd 
from scipy.stats import gamma, norm
from matplotlib import pyplot as plt
from subprocess import Popen, check_output, STDOUT

# A = call('gcloud compute instances list --filter="name=(jonavind)"', shell=True)
# A = Popen('gcloud compute instances list"', shell=True)
# print(A)
# output = check_output('gcloud compute instances list"',shell=True,stderr=STDOUT).decode()
# print('Here is the output:', output)
# print(output.split('\n')[10])

# ws = np.array([6,6,6,6,6,6])
# ws = np.array([
#     [1,2],
#     [3,4],
#     [5,6],
#     [7,8],
# ])

# alphas = np.array([1,1,2,3,4,5])

# A = norm.pdf(ws,0,alphas)
# print(A)

#     def procrastination():
#         from mpl_toolkits.mplot3d import Axes3D
#         def axisEqual3D(ax):
#             extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
#             sz = extents[:,1] - extents[:,0]
#             centers = np.mean(extents, axis=1)
#             maxsize = max(abs(sz))
#             r = maxsize/2
#             for ctr, dim in zip(centers, 'xyz'):
#                 getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

#         P = PolynomialFeatures(degree=2)
#         X = np.random.randn(100,2)
#         P.fit(X)
#         print(X)

#         print(P.get_feature_names(['x','y','z']))

#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')

#         f = lambda x,y: x**2+y**2

#         TT, RR = np.meshgrid(np.linspace(0,2*np.pi,20), np.linspace(0,3,20))
#         XX = (RR*np.cos(TT)).T
#         YY = (RR*np.sin(TT)).T
#         X = np.array([XX.ravel(), YY.ravel()]).T

#         z = f(*X.T)

#         x_gen = np.random.randn(100,2)
#         ax.scatter(*x_gen.T, f(*x_gen.T))
#         ax.plot_surface(XX, YY, z.reshape(XX.shape), alpha=0.2)
#         ax.plot_surface(XX, YY, np.zeros_like(XX), alpha=0.2)
#         axisEqual3D(ax)
#         plt.show()

    # procrastination()

    # def procrastiantion2():
    #     from mpl_toolkits.mplot3d import Axes3D
        
    # polynomials=[
    #     '69x+420',
    #     '30x^2+42x-420',
    #     '-x^3+42x^2-20x+69',
    #     '69x+69y-420',
    #     'xy-x^2+y^2-420',
    #     '2x^3-y^3-3xy^2+3x^2y+x^3-3yx+69',
    # ]

