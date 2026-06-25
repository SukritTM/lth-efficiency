import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader # for type hinting only lol 

def train_step_stream(models: nn.Module, X: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module, optimizers: torch.optim.Optimizer, device=None):
    if device is None:
        device = X.device

    X = X.to(device)
    y = y.to(device)
    streams = [torch.cuda.Stream() for _ in models]

    streams = [torch.cuda.Stream() for _ in models]

    for model, optimizer, stream in zip(models, optimizers, streams):
        
        with torch.cuda.stream(stream):
            optimizer.zero_grad()
            y_pred = model(X)
            # print(model(X))

            # loss = loss_fn(y_pred, y)
            yhot = F.one_hot(y, 10).to(torch.float32)
            loss = loss_fn(y_pred, yhot)

            loss.backward()
            optimizer.step()

def train_epoch_stream(models, train_loader, loss_fn, optimizers, device=None):
    if device is None:
        device = next(models.parameters()).device
    for model in models: model.train()

    for batch, (X, y) in enumerate(train_loader):
        train_step_stream(models, X, y, loss_fn, optimizers, device=device)

def train_loop_stream(models, train_loader, loss_fn, optimizers, n_epochs, device=None):
    if device is None:
        device = next(models[0].parameters()).device

    for epoch in range(n_epochs):
        train_epoch_stream(models, train_loader, loss_fn, optimizers, device=device)