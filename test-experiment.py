import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from LTH.datasets import get_mnist_dataset, get_loaders
from LTH.traineval import train_loop, evaluate_model, evaluate_model_loader
from LTH.models import construct_mlp

import pickle
import argparse

tseed = 468746545231360+1
nseed = 6543541+1
torch.manual_seed(tseed)
np.random.seed(nseed)

results = {}
results['seeds'] = {'torch': tseed, 'numpy': nseed}

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs')
parser.add_argument('-s', '--hidden-size', default='32')
parser.add_argument('-d', '--device', default='cpu')

arguments = parser.parse_args()
print(arguments)

assert arguments.device in ['cpu', 'cuda'], 'Device must be either cpu or cuda'
DEVICE = torch.device(arguments.device if torch.cuda.is_available() else 'cpu')

EPOCHS = int(arguments.epochs)
hidden_size = int(arguments.hidden_size)

train_set, test_set = get_mnist_dataset()
train_loader, test_loader = get_loaders(train_set, test_set, batch_size=128)

model = construct_mlp([784, hidden_size, 10], flatten_input=True)
model.to(device=DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loop(model, train_loader, loss_fn, optimizer, EPOCHS, silent=False, device=DEVICE)
train_loss, train_acc = evaluate_model(model, train_set.data.to(torch.float32)/255.0, train_set.targets, loss_fn, device=DEVICE)
test_loss, test_acc = evaluate_model(model, test_set.data.to(torch.float32)/255.0, test_set.targets, loss_fn, device=DEVICE)

print(f'train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}')
print(f'test_loss:  {test_loss:.4f} | test_acc:  {test_acc:.4f}')

results['FC-train-loss'] = train_loss
results['FC-train-acc']  = train_acc
results['FC-test-loss']  = test_loss
results['FC-test-acc']   = test_acc

with open(f'experiment_data/test-data-ss-e{EPOCHS}-s{hidden_size}.pkl', 'wb') as f:
    pickle.dump(results, f)