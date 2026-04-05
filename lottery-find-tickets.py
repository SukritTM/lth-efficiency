import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from LTH.datasets import get_mnist_dataset, get_loaders
from LTH.traineval import train_loop, evaluate_model, evaluate_model_loader
from LTH.models import construct_mlp
from LTH.models import PrunableModel

import os
import pickle
import argparse
from time import perf_counter as pf


tseed = 468746545260+1
nseed = 65431+1
torch.manual_seed(tseed)
np.random.seed(nseed)

results = {}
results['seeds'] = {'torch': tseed, 'numpy': nseed}

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs')
parser.add_argument('-r', '--num-rounds')
parser.add_argument('-p', '--pruning-ratio')
parser.add_argument('-t', '--num-tickets', default='15')
parser.add_argument('-s', '--hidden-size', default='32')
parser.add_argument('-d', '--device')

arguments = parser.parse_args()
print(arguments)

assert arguments.device in ['cpu', 'cuda'], 'Device must be either cpu or cuda'
DEVICE = torch.device(arguments.device if torch.cuda.is_available() else 'cpu')

EPOCHS = int(arguments.epochs)
hidden_size = int(arguments.hidden_size)
pruning_ratio = float(arguments.pruning_ratio)
num_rounds = int(arguments.num_rounds)
NUM_TICKETS = int(arguments.num_tickets)

results['config'] = {
    'epochs': EPOCHS,
    'num_rounds': num_rounds,
    'pruning_ratio': pruning_ratio,
    'hidden_size': hidden_size,
}


def get_multiple_models(n_models, train_loader, test_loader, train_set, test_set, hidden_size=32, silent=True):

    if not silent:
        print('Finding base models...')

    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    models: list[PrunableModel] = []
    for i in range(n_models):
        timer = pf()

        model = construct_mlp([784, hidden_size, 10], flatten_input=True)
        prunable = PrunableModel(model, device=DEVICE)

        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(params=prunable.parameters(), lr=0.001)

        train_loop(prunable, train_loader, loss_fn, optimizer, EPOCHS, silent=True)
        train_loss, train_acc = evaluate_model(prunable, train_set.data.to(torch.float32)/255.0, train_set.targets, loss_fn)
        test_loss, test_acc = evaluate_model(prunable, test_set.data.to(torch.float32)/255.0, test_set.targets, loss_fn)

        prunable.to(device='cpu')
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        models.append(prunable)

        timer = pf() - timer
        if not silent: print(f'Model {i} trained, took {timer:0.2f} seconds', flush=True)

    print()
    return models, train_losses, train_accs, test_losses, test_accs


train_set, test_set = get_mnist_dataset()
train_loader, test_loader = get_loaders(train_set, test_set, batch_size=64, shuffle=False)

models, train_losses, train_accs, test_losses, test_accs = get_multiple_models(
    n_models     = NUM_TICKETS,
    train_loader = train_loader,
    test_loader  = test_loader,
    train_set    = train_set,
    test_set     = test_set,
    hidden_size  = hidden_size,
    silent       = False
)

results['FC-train-losses'] = train_losses
results['FC-train-accs']   = train_accs
results['FC-test-losses']  = test_losses
results['FC-test-accs']    = test_accs

loss_fn = nn.CrossEntropyLoss(reduction='mean')
n_epochs = EPOCHS

# ITERATIVE MAGNITUDE PRUNING
print('Beginning pruning...')
for i, prunable in enumerate(models):
    prunable.to(device=DEVICE)
    timer = pf()
    for r in range(num_rounds):
        prunable.apply_saved_initialization()
        new_optimizer = torch.optim.Adam(params=prunable.parameters(), lr=0.001)
        train_loop(prunable, train_loader, loss_fn, new_optimizer, n_epochs, silent=True)
        prunable.find_mask(pruning_ratio)

    # Winning ticket: reset to saved init with mask applied
    prunable.apply_saved_initialization()
    timer = pf() - timer
    prunable.to(device='cpu')
    print(f'Pruning {i} complete, took {timer:0.2f} seconds', flush=True)

print()

results['winning-tickets'] = [prunable.retrieve_pruned_initialization() for prunable in models]
results['winning-tickets-full-initializations'] = [prunable.retrieve_unpruned_initialization() for prunable in models]

cpumasks = []
for prunable in models:
    cpumask = {key: prunable.mask[key].cpu().clone().detach() for key in prunable.mask.keys()}
    cpumasks.append(cpumask)

results['winning-ticket-masks'] = cpumasks

# Evaluate pruned models before and after one final training run
results['pruned-train-loss-before'] = []
results['pruned-train-acc-before']  = []
results['pruned-test-loss-before']  = []
results['pruned-test-acc-before']   = []

results['pruned-train-loss-after'] = []
results['pruned-train-acc-after']  = []
results['pruned-test-loss-after']  = []
results['pruned-test-acc-after']   = []

print('Training pruned models...', end='', flush=True)
for prunable in models:
    prunable.to(device=DEVICE)

    train_loss, train_acc = evaluate_model(prunable, train_set.data.to(torch.float32)/255.0, train_set.targets, loss_fn)
    test_loss, test_acc = evaluate_model(prunable, test_set.data.to(torch.float32)/255.0, test_set.targets, loss_fn)

    results['pruned-train-loss-before'].append(train_loss)
    results['pruned-train-acc-before'].append(train_acc)
    results['pruned-test-loss-before'].append(test_loss)
    results['pruned-test-acc-before'].append(test_acc)

    new_optimizer = torch.optim.Adam(params=prunable.parameters(), lr=0.001)
    train_loop(prunable, train_loader, loss_fn, new_optimizer, n_epochs, silent=True)

    train_loss, train_acc = evaluate_model(prunable, train_set.data.to(torch.float32)/255.0, train_set.targets, loss_fn)
    test_loss, test_acc = evaluate_model(prunable, test_set.data.to(torch.float32)/255.0, test_set.targets, loss_fn)

    results['pruned-train-loss-after'].append(train_loss)
    results['pruned-train-acc-after'].append(train_acc)
    results['pruned-test-loss-after'].append(test_loss)
    results['pruned-test-acc-after'].append(test_acc)

    prunable.to(device='cpu')
print('done\n')

print('Subnetwork search complete.')
from pprint import pprint
print('Saving:')
pprint(list(results.keys()))

if not os.path.exists('experiment_data'):
    os.mkdir('experiment_data')

out_path = f'experiment_data/subnetworks-e{EPOCHS}-r{num_rounds}-p{pruning_ratio:0.4f}-t{NUM_TICKETS}-s{hidden_size}.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(results, f)

print(f'\nSaved to: {out_path}')