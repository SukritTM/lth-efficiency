import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader # for type hinting only lol 


def train_step(model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device=None):
    if device is None:
        device = X.device

    X = X.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    y_pred = model(X)
    # print(model(X))

    # loss = loss_fn(y_pred, y)
    yhot = F.one_hot(y, 10).to(torch.float32)
    loss = loss_fn(y_pred, yhot)

    loss.backward()
    optimizer.step()

    return loss

def train_epoch(model, train_loader, loss_fn, optimizer, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.train()

    total_loss = torch.tensor(0.0, device=device)
    for batch, (X, y) in enumerate(train_loader):
        batch_loss = train_step(model, X, y, loss_fn, optimizer, device=device)
        total_loss += batch_loss.item()

    return total_loss / (batch + 1)

def train_loop(model, train_loader, loss_fn, optimizer, n_epochs, silent=False, device=None):
    if device is None:
        device = next(model.parameters()).device

    for epoch in range(n_epochs):
        epoch_loss = train_epoch(model, train_loader, loss_fn, optimizer, device=device)

        if not silent:
            print(f'Epoch {epoch+1} | train_loss: {epoch_loss.item()}')

def evaluate_model(model, X, y, loss_fn, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    X = X.to(device)
    y = y.to(device)
    with torch.no_grad():
        y_pred = model(X)
        loss = loss_fn(y_pred, F.one_hot(y, 10).to(torch.float32))
        y_pred = torch.argmax(y_pred, -1)
        acc = torch.sum(y_pred == y) / y.size()[0]

    return loss.cpu().item(), acc.cpu().item()

def evaluate_model_loader(model, loader: DataLoader, loss_fn, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    n_samples = 0
    
    with torch.no_grad():
        n_correct = torch.tensor(0, device=device)
        loss = torch.tensor(0, dtype=torch.float32, device=device)
        for batch, (X, y) in enumerate(loader):
            X = X.to(device)
            y = y.to(device)
            n_samples += len(y)

            y_pred = model(X)
            loss += loss_fn(y_pred, F.one_hot(y, 10).to(torch.float32)) * len(y)
            y_pred = torch.argmax(y_pred, -1)
            n_correct += torch.sum(y_pred == y)

        loss /= n_samples
        acc = n_correct / n_samples

    return loss.cpu().item(), acc.cpu().item()