import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns;

sns.set()
from tst import Transformer
from models.custom_transformer import CustomTransformer

BATCH_SIZE = 128
NUM_WORKERS = 0
LR = 1e-5
EPOCHS = 50
attention_size = 24  # Attention window size
dropout = 0.2  # Dropout rate
pe = 'original'  # Positional encoding
chunk_mode = None

K = 10  # Time window length
d_model = 48  # Lattent dim
q = 8  # Query size
v = 8  # Value size
h = 4  # Number of heads
N = 4  # Number of encoder and decoder to stack
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
from util import getdata, getdata_energy_after_Corona

WINDOW = 10
# data = np.random.random_sample((100, 15, 12))
# data = getdata(window=WINDOW)
data = getdata_energy_after_Corona(window=WINDOW)
d_input = np.shape(data)[-1] - 1  # From dataset
d_output = 1  # From dataset

# data = data[:300]
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
# print(data.shape)
# print(data != data)
# print((data != data).any())

train_dataloader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS
                              )

val_data = DataLoader(val_data,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=NUM_WORKERS
                      )

all_data = DataLoader(all_data,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=NUM_WORKERS
                      )
# Load transformer with Adam optimizer and MSE loss function
model_class = CustomTransformer
net = model_class(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size, dropout=dropout,
                  chunk_mode=chunk_mode, pe=pe).to(device)
clip_value = 0.5
for p in net.parameters():
    if p.requires_grad:
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
optimizer = optim.Adam(net.parameters(), lr=LR)
loss_function = nn.MSELoss()

# Prepare loss history
hist_loss = np.zeros(EPOCHS)
for idx_epoch in range(EPOCHS):
    running_loss = 0
    with tqdm(total=len(train_dataloader.dataset), desc=f"[Epoch {idx_epoch+1:3d}/{EPOCHS}]") as pbar:
        for idx_batch, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = batch[:, :, :-1]
            y = batch[:, -1, -1]

            netout = net(x)
            netout = netout[:, -1, -1]
            y = y.reshape(-1, 1)
            netout = netout.reshape(-1, 1)
            # Comupte loss
            loss = loss_function(netout, y)

            # Backpropage loss
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (idx_batch + 1)})
            pbar.update(BATCH_SIZE)

    hist_loss[idx_epoch] = running_loss / len(train_dataloader)
# plt.plot(hist_loss, 'o-')
print(f"Loss: {float(hist_loss[-1]):5f}")

## Select training example
y_pred = []
y_true = []
for batch in all_data:
    x = batch[:, :, :-1]
    y = batch[:, -1, -1]
    # x = x.reshape(1, 15, 11)
    # Run predictions
    with torch.no_grad():
        netout = net(x)

    # print(netout.shape)
    # print(y.shape)

    idx_output_var = d_input
    # Select real temperature
    true = y.cpu().numpy()
    y_true.extend(true)
    pred = netout[:, -1, -1]
    pred = pred.cpu().numpy()
    # print(pred)
    y_pred.extend(pred)
from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_true, y_pred))
from util import plot_result

plot_result(y_pred, y_true)
