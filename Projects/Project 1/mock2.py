from scipy.stats import gamma
import numpy as np 
from matplotlib import pyplot as plt 

x = np.linspace(0,10,100)

plt.plot(x, gamma.pdf(x, 1.1, scale=1/0.25))
plt.show()