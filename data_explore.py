import numpy as np
import xlrd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from util import all_energy_data,energy_return_data
from scipy.ndimage.filters import uniform_filter,uniform_filter1d
_, np_data = energy_return_data(window=10)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# new_one = []
# for i in range(7):
#     np_data[:,i]= uniform_filter1d( np_data[:,i],size=10, mode='constant')
# np_data = uniform_filter1d(np_data,size=20, mode='constant')
xx = np.array(range(len(np_data)))
plt.plot(xx, np_data[:, 0], label="Return")
plt.plot(xx, np_data[:, 1], label="EU/USD")
# plt.plot(xx, np_data[:, 3], label="CO")
plt.plot(xx, np_data[:, 2], label="NS")
plt.plot(xx, np_data[:, 3], label="USD Index")
plt.plot(xx, np_data[:, 4], label="CO")
# plt.plot(xx, np_data[:, 0], label="price")
plt.title('Explore')
plt.legend(loc="upper left")

plt.show()
