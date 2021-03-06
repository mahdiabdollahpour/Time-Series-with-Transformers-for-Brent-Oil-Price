import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.cuda import device
from torch.utils.data import DataLoader
from tqdm import tqdm
# import seaborn as sns;

# sns.set()
from tst import Transformer
from models.custom_transformer import CustomTransformer
from models.Mov_transformer import MoveTransformer
from models.transformer_lstm import TransformerLSTM
from sklearn.metrics import mean_squared_error

from util import plot_result, plot_loss

save_path = 'saved/transformer_checkpoint.pth'
BATCH_SIZE = 20
LR = 5e-5
EPOCHS = 35
WINDOW = 25
attention_size = 16  # Attention window size
dropout = 0.2  # Dropout rate
pe = 'regular'  # Positional encoding
chunk_mode = None

# K = 10  # Time window length
d_model = 20  # Lattent dim
q = 8  # Query size
v = 8  # Value size
h = 4  # Number of heads
N = 2  # Number of encoder and decoder to stack
device = torch.device( "cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
from util import energy_return_movement_data, get_dataloaders

train_dataloader, val_data, test_data, all_data, d_input, d_output, scaler = get_dataloaders(WINDOW,
                                                                                             energy_return_movement_data,
                                                                                             BATCH_SIZE, device,
                                                                                             shuffle=True)
# data = np.random.random_sample((100, 15, 12))
# data = getdata(window=WINDOW)
# Load transformer with Adam optimizer and MSE loss function
arguments = {
    'd_input': d_input, 'd_model': d_model, 'd_output': d_output, 'q': q, 'v': v,
    'h': h, 'N': N, 'attention_size': attention_size, 'dropout': dropout,
    'chunk_mode': chunk_mode, 'pe': pe,
}
from models.helpers import get_model

# model_class = TransformerLSTM
# model_class = CustomTransformer
model_class = MoveTransformer
# model_class = Transformer
net, optimizer = get_model(model_class, arguments, device, lr=LR)
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(total_params)
# Prepare loss history
from models.helpers import compute_loss

loss_function = nn.CrossEntropyLoss()
hist_loss = np.zeros(EPOCHS)
val_losses = []
running_losses = []
with torch.autograd.set_detect_anomaly(True):
    for idx_epoch in range(EPOCHS):
        running_loss = 0
        with tqdm(total=len(train_dataloader.dataset), desc=f"[Epoch {idx_epoch+1:3d}/{EPOCHS}]") as pbar:
            for idx_batch, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                x = batch[:, :, :-d_output]
                y = batch[:, -1, -d_output]

                netout = net(x)
                netout = netout[:, -1, -d_output]
                y = y.type(torch.DoubleTensor).reshape(-1, 1)
                netout = netout.reshape(-1, 1)

                print(y.shape,netout.shape)
                # Comupte loss
                loss = loss_function(netout, y)

                # Backpropage loss
                loss.backward()

                # Update weights
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (idx_batch + 1)})
                pbar.update(BATCH_SIZE)
        # val_loss = compute_loss(net, val_data, mean_squared_error)
        # train_loss = compute_loss(net, train_dataloader, mean_squared_error)
        # val_losses.append(val_loss)
        # running_losses.append(train_loss)
        hist_loss[idx_epoch] = running_loss / len(train_dataloader)


# plt.plot(hist_loss, 'o-')
# plot_loss(val_losses, running_losses)


def analzie(data_loader):
    ## Select training example
    y_pred = []
    y_true = []
    for batch in data_loader:
        x = batch[:, :, :-1]
        y = batch[:, -1, -1]
        with torch.no_grad():
            netout = net(x)
        true = y.cpu().numpy()
        y_true.extend(true)
        pred = netout[:, -1, -1]
        pred = pred.cpu().numpy()
        # print(pred)
        y_pred.extend(pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.heaviside(y_pred,0)
    print(np.mean(np.abs(y_pred - y_true)))
    # np.save('transformer',y_pred)
    # np.save('ture',y_true)
    # orig_y_true = scaler.inverse_transform(y_true)
    # orig_y_pred = scaler.inverse_transform(y_pred)
    # full_report(y_true, y_pred)
    # plot_result(y_true, y_pred)


analzie(all_data)
# analzie(val_data)
analzie(test_data)

# checkpoint = {
#     'model_state_dict': net.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'args': arguments,
#     'lr': LR
# }
#
# torch.save(checkpoint, save_path)
