import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-4,4,100)
y1 = x**2
y2 = 1-x**2

dy1 = 2*x
dy2 = -2*x

plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, dy1)
plt.plot(x, dy2)
plt.show()