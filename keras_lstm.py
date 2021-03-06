import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np

WINDOW = 25
BATCH_SIZE = 64
epochs = 25
lr = 0.0005

from util import getdata
from sklearn.metrics import mean_squared_error
from util import getdata, getdata_energy_after_Corona, get_dataloaders, all_energy_data, full_report, energy_return_data

data, _, scaler = energy_return_data(window=WINDOW)
print(np.shape(data))
d_input = np.shape(data)[-1] - 1  # From dataset
d_output = 1  # From dataset

# data = data[:300]
data = data.astype(np.float32)
n_data = len(data)

train_data = data[:int(0.9 * n_data)]
# val_data = data[int(0.8 * n_data):int(0.9 * n_data)]
test_data = data[int(0.9 * n_data):]

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(20, return_sequences=True),

    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

lstm_model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(lr)
)
lstm_model.fit(train_data[:, :, :-1], train_data[:, :, -1], validation_split=0.1, epochs=epochs, batch_size=BATCH_SIZE,
               verbose=2)
print(lstm_model.summary())

from util import plot_result, plot_loss


def analzie(data_set,save=False):
    ## Select training example

    y_true = data_set[:, -1, -1]
    predictions = lstm_model.predict(data_set[:, :, :-1])
    y_pred = predictions[:, -1, -1]
    if save:
        np.save('lstm', y_pred)
    full_report(y_true, y_pred)
    plot_result(y_true, y_pred)


analzie(data)
# analzie(val_data)
analzie(test_data)
