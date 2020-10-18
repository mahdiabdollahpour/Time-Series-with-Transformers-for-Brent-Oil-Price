import torch
import torch.optim as optim

import torch.nn as nn


def get_model(model_class, arguments, device, lr):
    net = model_class(**arguments).to(device)
    clip_value = 0.5
    for p in net.parameters():
        if p.requires_grad:
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    optimizer = optim.Adam(net.parameters(), lr=lr)

    return net, optimizer


def compute_loss(net, dataloader, metric):
    y_pred = []
    y_true = []
    for batch in dataloader:
        x = batch[:, :, :-1]
        y = batch[:, -1, -1]
        with torch.no_grad():
            netout = net(x)

        # Select real temperature
        true = y.cpu().numpy()
        y_true.extend(true)
        pred = netout[:, -1, -1]
        pred = pred.cpu().numpy()
        # print(pred)
        y_pred.extend(pred)

    return metric(y_true, y_pred)
