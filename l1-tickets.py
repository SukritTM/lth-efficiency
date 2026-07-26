import torch
from torch import nn
 
import numpy as np
 
from LTH.datasets import get_mnist_dataset, get_loaders
from LTH.traineval import train_loop, evaluate_model
from LTH.traineval_stream import train_loop_l1_stream, train_loop_stream
from LTH.models import construct_mlp
from LTH.models import PrunableModel
 
import os
import pickle
import argparse
from time import perf_counter as pf

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--filepath') 
parser.add_argument('-r', '--remove-fraction')
parser.add_argument('-l', '--loss', default='auto', choices=['mse', 'crossentropy', 'auto'], help='"mse", "crossentropy", or "auto". "auto" uses the value in the config, if it exists, else defaults to crossentropy.')
parser.add_argument('-d', '--device')

arguments = parser.parse_args()
print(arguments)

with open(arguments.filepath, 'rb') as f:
    network_data = pickle.load(f)

cfg: dict   = network_data['config']
EPOCHS      = int(cfg['epochs'])
hidden_size = int(cfg['hidden_size'])
NUM_TICKETS = len(network_data['model-initializations'])
DEVICE      = torch.device(arguments.device)

remove_fraction = float(arguments.remove_fraction)
loss_fn_type = arguments.loss

models: list[PrunableModel] = []
optimizers: list[torch.optim.Optimizer] = []
for idx in range(NUM_TICKETS):
    model = construct_mlp([784, hidden_size, 10], flatten_input=True)
 
    # PrunableModel.__init__ calls reinitialize_randomly() then saves that state.
    # We immediately overwrite both below, so the device here is just 'cpu' for setup.
    prunable = PrunableModel(model)
 
    # Restore the full (unpruned) initialization so apply_saved_initialization()
    # and any future retrieve_*() calls behave correctly.
    full_init = network_data['model-initializations'][idx]
    prunable.saved_initialization = {
        key: tensor.clone().detach().cpu() for key, tensor in full_init.items()
    }
    prunable.apply_saved_initialization()
    prunable.find_mask(remove_fraction)
    # Restore mask (already CPU tensors from the pkl)
    # prunable.mask = {
    #     key: tensor.clone().detach().cpu()
    #     for key, tensor in subnetwork_data['winning-ticket-masks'][idx].items()
    # }
 
    # Set model weights to winning-ticket init with mask applied
    prunable.apply_saved_initialization()
    prunable.to(DEVICE)   
    models.append(prunable)

    optim = torch.optim.Adam(prunable.parameters(), lr=0.001)
    optimizers.append(optim)

train_set, test_set = get_mnist_dataset()
train_loader, test_loader = get_loaders(train_set, test_set, batch_size=64, shuffle=False)

results = {}
for key, val in network_data.items():
    results[key] = val

results['config']['remove_fraction'] = remove_fraction
results['config']['ticket-loss-choice'] = loss_fn_type

if loss_fn_type == 'auto':
    parent_type = cfg.get('loss')
    if parent_type == None:
        loss_fn_type = 'crossentropy'
        print('WARNING: loss is specified as "auto" but the input file does not specify a loss function. Defaulting to crossentropy loss.')
    else:
        loss_fn_type = parent_type

if loss_fn_type == 'crossentropy': loss_fn = nn.CrossEntropyLoss(reduction='mean')
elif loss_fn_type == 'mse': loss_fn = nn.MSELoss(reduction='mean')
else: raise Exception('Loss fn type was neither "mse" nor "crossentropy"')

train_losses, train_accs, test_losses, test_accs = [], [], [], []
for i in range(NUM_TICKETS):
    train_loss, train_acc = evaluate_model(models[i], train_set.data.to(torch.float32)/255.0, train_set.targets, loss_fn)
    test_loss, test_acc = evaluate_model(models[i], test_set.data.to(torch.float32)/255.0, test_set.targets, loss_fn)

    models[i].to(device='cpu')
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)


results['pruned-train-losses-before'] = train_losses
results['pruned-train-accs-before']   = train_accs
results['pruned-test-losses-before']  = test_losses
results['pruned-test-accs-before']    = test_accs

timer = pf()
train_loop_stream(
    models=models,
    train_loader=train_loader,
    loss_fn=loss_fn,
    optimizers=optimizers,
    n_epochs=EPOCHS
)

train_losses, train_accs, test_losses, test_accs = [], [], [], []
for i in range(NUM_TICKETS):
    train_loss, train_acc = evaluate_model(models[i], train_set.data.to(torch.float32)/255.0, train_set.targets, loss_fn)
    test_loss, test_acc = evaluate_model(models[i], test_set.data.to(torch.float32)/255.0, test_set.targets, loss_fn)

    models[i].to(device='cpu')
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

cpuweights_tickets = []
for prunable in models:
    prunable._apply_mask()
    cpuweight = {name: param.cpu().clone().detach() for name, param in prunable.named_parameters()}
    cpuweights_tickets.append(cpuweight)
results['ticket-params'] = cpuweights_tickets

timer = pf() - timer
print(f'{NUM_TICKETS} tickets trained in {timer:0.2f} seconds', flush=True)

results['pruned-train-losses-after'] = train_losses
results['pruned-train-accs-after']   = train_accs
results['pruned-test-losses-after']  = test_losses
results['pruned-test-accs-after']    = test_accs

results['input-file-keys'] = network_data

if not os.path.exists('experiment_data'):
    os.mkdir('experiment_data')

out_path = f'experiment_data/l1-tickets-e{EPOCHS}-t{NUM_TICKETS}-s{hidden_size}-r{remove_fraction:.4f}.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(results, f)

print(f'\nSaved to: {out_path}')