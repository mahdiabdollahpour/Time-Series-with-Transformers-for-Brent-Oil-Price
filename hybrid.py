from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from util import getdata_energy_after_Corona, plot_result, all_energy_data, full_report, energy_return_data

window = 25
true = np.load('true.npy')

lstm = np.load('lstm.npy')
transformer = np.load('transformer.npy')
mlpreg = np.load('mlpreg.npy')
svr = np.load('svr.npy')
GPreg = np.load('GPreg.npy')

lstm = lstm[5:]
transformer = transformer[5:]
mlpreg = mlpreg[5:]
svr = svr[5:]
GPreg = GPreg[5:]

lagged_5 = true[:-5]
lagged_4 = true[1:-4]
lagged_3 = true[2:-3]
lagged_2 = true[3:-2]
lagged_1 = true[4:-1]

true= true[5:]

print(true.shape)
print(lstm.shape)
print(transformer.shape)
print(mlpreg.shape)
print(svr.shape)
print(lagged_1.shape)
print(lagged_2.shape)
print(lagged_3.shape)
print(lagged_4.shape)
print(lagged_5.shape)

# data = np.array(list(zip(lstm, transformer, mlpreg, svr, GPreg, true)))
data = np.concatenate(
    (lstm.reshape(-1, 1), transformer.reshape(-1, 1), mlpreg.reshape(-1, 1), svr.reshape(-1, 1), GPreg.reshape(-1, 1),
     lagged_5.reshape(-1, 1), lagged_4.reshape(-1, 1), lagged_3.reshape(-1, 1), lagged_2.reshape(-1, 1),
     lagged_1.reshape(-1, 1),
     true.reshape(-1, 1)), axis=1)
print(data[0])
print(data.shape)

# data,_ = getdata_energy_after_Corona(window=10)
# data, _, scaler = energy_return_data(window=window)
# data = getdata_energy_after_Corona(window=10)
# n_data = len(np_data)
#     windowed = []
#     for i in range(n_data - window + 1):
#         seq = []
#         for j in range(window):
#             # print(i+j,len(data),n_data,window,n_data-window)
#             seq.append(np_data[i + j])
#         windowed.append(seq)
#     windowed = np.array(windowed)

X = data[:, :-1]

y = data[:, -1]
tr_size = int(0.9 * len(X))
X_tr = X[:tr_size]
y_tr = y[:tr_size]
X_test = X[tr_size:]
y_test = y[tr_size:]
print(np.shape(X))
print(np.shape(y))
rng = np.random.RandomState(0)

regr = MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes=(2))
regr.fit(X_tr, y_tr)

from sklearn.metrics import mean_squared_error


def analize(XX, yy):
    y_pred = regr.predict(XX)
    plot_result(yy, y_pred)
    full_report(yy, y_pred)


analize(X, y)
analize(X_test, y_test)
