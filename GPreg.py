from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

kernel = DotProduct() + WhiteKernel()
import numpy as np
from util import getdata_energy_after_Corona, plot_result, all_energy_data, full_report, energy_return_data

# data = getdata_energy_after_Corona(window=10)
data, _,scaler = energy_return_data(window=10)
X = data[:, :, :-1]
a, b, c = np.shape(X)
X = np.reshape(X, (-1, b * c))
y = data[:, -1, -1]
tr_size = int(0.9 * len(X))
X_tr = X[:tr_size]
y_tr = y[:tr_size]
X_test = X[tr_size:]
y_test = y[tr_size:]
print(np.shape(X))
print(np.shape(y))
rng = np.random.RandomState(0)

regr = GaussianProcessRegressor(kernel=kernel,
                                random_state=0)
regr.fit(X_tr, y_tr)

from sklearn.metrics import mean_squared_error

def analize(XX,yy):

    y_pred = regr.predict(XX)
    # orig_y_true = scaler.inverse_transform(yy)
    # orig_y_pred = scaler.inverse_transform(y_pred)
    plot_result(yy, y_pred)
    full_report(yy, y_pred)

analize(X,y)
analize(X_test,y_test)
