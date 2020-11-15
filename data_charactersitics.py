from util import energy_return_data
import numpy as np
from scipy.stats import kurtosis, skew
from statsmodels.stats.diagnostic import het_arch,acorr_ljungbox
windowed, np_data, scaler1 = energy_return_data(window=25)

y = np_data[:, :-1].reshape(-1)
print('mean', np.mean(y))
print('max', np.max(y))
print('min', np.min(y))
print('median', np.median(y))
print('S.D.', np.std(y))
print('Skewness', skew(y))
print('Excess kurtosis', kurtosis(y) - 3)
print('Ljung-Box test of autocorrelation in residuals', acorr_ljungbox(y,lags=[25]) )
print('Engle’s Test for Autoregressive Conditional Heteroscedasticity (ARCH)', het_arch(y))
