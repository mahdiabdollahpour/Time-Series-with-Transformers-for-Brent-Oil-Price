import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")


class LSTM(nn.Module):
    def __init__(self, input_size=1, timestep=30, BATCH_SIZE=64, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, timestep, self.hidden_layer_size, device=device),
                            torch.zeros(1, timestep, self.hidden_layer_size, device=device))

    def forward(self, input_seq):
        seq_len = input_seq.shape[1]
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), seq_len, -1), self.hidden_cell)
        last_time_step = \
            lstm_out.view(seq_len, len(input_seq), self.hidden_layer_size)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred


WINDOW = 30
BATCH_SIZE = 64
epochs = 500
lr = 0.00001
model = LSTM(input_size=11, timestep=WINDOW, BATCH_SIZE=BATCH_SIZE)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.cuda()
from util import getdata

# data = np.random.random_sample((100, 15, 12))
data = getdata(window=WINDOW)
data = data[:50]
data = data.astype(np.float32)
n_data = len(data)
train_data = data[:int(0.7 * n_data)]
val_data = data[int(0.7 * n_data):int(0.9 * n_data)]
test_data = data[int(0.9 * n_data):]
# data = torch.from_numpy(data,device=device)
all_data = torch.tensor(data, device=device)
train_data = torch.tensor(train_data, device=device)
val_data = torch.tensor(val_data, device=device)
test_data = torch.tensor(test_data, device=device)

train_dataloader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True,

                              )

val_data = DataLoader(val_data,
                      batch_size=BATCH_SIZE,
                      shuffle=True,

                      )

all_data = DataLoader(all_data,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      )


for i in range(epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        x = batch[:, :, :-1]
        y = batch[:, -1, -1]

        seq_len = x.shape[1]

        model.hidden_cell = (torch.zeros(1, seq_len, model.hidden_layer_size, device=device),
                             torch.zeros(1, seq_len, model.hidden_layer_size, device=device))

        # print(x.shape, seq_len)
        y_pred = model(x)
        # print(y_pred.shape, y.shape)
        single_loss = loss_function(y_pred, y)
        single_loss.backward()
        optimizer.step()

    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


y_pred = []
y_true = []
for batch in all_data:
    optimizer.zero_grad()
    x = batch[:, :, :-1]
    y = batch[:, -1, -1]

    seq_len = x.shape[1]

    model.hidden_cell = (torch.zeros(1, seq_len, model.hidden_layer_size, device=device),
                         torch.zeros(1, seq_len, model.hidden_layer_size, device=device))
    # Run predictions
    with torch.no_grad():
        netout = model(x)


    # Select real temperature
    true = y.cpu().numpy()
    y_true.extend(true)
    pred = netout.cpu().numpy()
    # print(pred)
    y_pred.extend(pred)

from utils import plot_result

plot_result(y_pred, y_true)
