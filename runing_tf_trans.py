import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.disable_eager_execution()
tf.enable_resource_variables()
tf.enable_v2_tensorshape()
tf.enable_control_flow_v2()
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm

sns.set()
tf.random.set_random_seed(1234)
from models.tf_transformer import *

num_layers = 1
size_layer = 128
timestamp = 5

dropout_rate = 0.8
# future_day = test_size
learning_rate = 0.001
# WINDOW = 30
BATCH_SIZE = 64
epochs = 30

from util import getdata
from sklearn.metrics import mean_squared_error
from util import getdata, getdata_energy_after_Corona, get_dataloaders, all_energy_data, full_report, energy_return_data

data, _, scaler = energy_return_data(window=timestamp)
d_input = np.shape(data)[-1] - 1  # From dataset
d_output = 1  # From dataset

# data = data[:300]
data = data.astype(np.float32)
n_data = len(data)
train_data = data[:int(0.9 * n_data)]
# val_data = data[int(0.8 * n_data):int(0.9 * n_data)]
test_data = data[int(0.9 * n_data):]

import math


def forecast():
    tf.reset_default_graph()
    modelnn = Attention(size_layer, size_layer, learning_rate, d_input, d_output)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

    pbar = tqdm(range(epochs), desc='train loop')
    number_of_batches = math.ceil(len(train_data) / BATCH_SIZE)
    print('There are', number_of_batches, 'batches in each epoch')

    # all_x = tf.data.Dataset.from_tensor_slices(train_data[:, -1])
    # all_y = tf.data.Dataset.from_tensor_slices(train_data[:, -1])
    # train_dataset = tf.data.Dataset.zip((all_x, all_y))
    # batched_dataset = train_dataset.batch(BATCH_SIZE)
    for i in pbar:
        total_loss, total_acc = [], []
        for j in range(number_of_batches):
            bs = min(len(train_data) - j * BATCH_SIZE, BATCH_SIZE)

            batch_x = train_data[j * BATCH_SIZE:j * BATCH_SIZE + bs, :, :-1]
            batch_y = train_data[j * BATCH_SIZE:j * BATCH_SIZE + bs, -1, -1]
            batch_y = batch_y.reshape(bs, 1)
            logits, _, loss = sess.run(
                [modelnn.logits, modelnn.optimizer, modelnn.cost],
                feed_dict={
                    modelnn.X: batch_x,
                    modelnn.Y: batch_y,
                },
            )
            total_loss.append(loss)
            total_acc.append(calculate_accuracy(y[:, 0], logits[:, 0]))

        pbar.set_postfix(cost=np.mean(total_loss), acc=np.mean(total_acc))

    test_x = tf.data.Dataset.from_tensor_slices(test_data[:, -1])
    # test_y = tf.data.Dataset.from_tensor_slices(test_data[:, -1])
    # test_dataset = tf.data.Dataset.zip(all_x, all_y)
    # test_batched_dataset = test_dataset.batch(BATCH_SIZE)

    out_logits = sess.run(
        modelnn.logits,
        feed_dict={
            modelnn.X: test_x
        },
    )
    print(np.shape(out_logits))
    # deep_future = anchor(output_predict[:, 0], 0.3)


forecast()
