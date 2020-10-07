import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns;

sns.set()
from tst import Transformer
from dataset import OzeDataset

BATCH_SIZE = 4
NUM_WORKERS = 0
LR = 1e-2
EPOCHS = 2
attention_size = 24  # Attention window size
dropout = 0.2  # Dropout rate
pe = None  # Positional encoding
chunk_mode = None

K = 300  # Time window length
d_model = 48  # Lattent dim
q = 8  # Query size
v = 8  # Value size
h = 4  # Number of heads
N = 4  # Number of encoder and decoder to stack
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

d_input = 11  # From dataset
d_output = 1  # From dataset
from excel_to_oze import getdata
data = np.random.random_sample((300,12))
# data = getdata()
data = data.astype(np.float32)
# data = torch.from_numpy(data,device=device)
data = torch.tensor(data, device=device)
print(data.shape)
dataloader = DataLoader(data,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS
                        )

# Load transformer with Adam optimizer and MSE loss function
net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size, dropout=dropout,
                  chunk_mode=chunk_mode, pe=pe).to(device)

optimizer = optim.Adam(net.parameters(), lr=LR)
loss_function = nn.MSELoss()

# Prepare loss history
hist_loss = np.zeros(EPOCHS)
for idx_epoch in range(EPOCHS):
    running_loss = 0
    with tqdm(total=len(dataloader.dataset), desc=f"[Epoch {idx_epoch+1:3d}/{EPOCHS}]") as pbar:
        for idx_batch, batch in enumerate(dataloader):
            optimizer.zero_grad()
            x = batch[:, :-1]
            y = batch[:, -1]
            print(x)
            # print(y)
            # Propagate input
            netout = net(x)

            # Comupte loss
            loss = loss_function(netout, y)

            # Backpropage loss
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (idx_batch + 1)})
            pbar.update(BATCH_SIZE)

    hist_loss[idx_epoch] = running_loss / len(dataloader)
plt.plot(hist_loss, 'o-')
print(f"Loss: {float(hist_loss[-1]):5f}")

## Select training example
idx = np.random.randint(0, len(dataloader.dataset))
x, y = dataloader.dataset[idx]

# Run predictions
with torch.no_grad():
    x = torch.Tensor(x[np.newaxis, ...])
    netout = net(x)

plt.figure(figsize=(30, 30))
idx_output_var = d_input
# Select real temperature
y_true = y[:, idx_output_var]

y_pred = netout[0, :, idx_output_var]
y_pred = y_pred.numpy()

plt.subplot(8, 1, idx_output_var + 1)

plt.plot(y_true, label="Truth")
plt.plot(y_pred, label="Prediction")
plt.title('output')
plt.legend()
plt.savefig("fig.jpg")

# Select first encoding layer
encoder = net.layers_encoding[0]

# Get the first attention map
attn_map = encoder.attention_map[0]

# Plot
plt.figure(figsize=(20, 20))
sns.heatmap(attn_map)
plt.savefig("attention_map.jpg")
