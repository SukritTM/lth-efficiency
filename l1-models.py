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

tseed = 468746545260+1
nseed = 65431+1
torch.manual_seed(tseed)
np.random.seed(nseed)

results = {}
results['seeds'] = {'torch': tseed, 'numpy': nseed}

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs')
parser.add_argument('-t', '--num-tickets', default='15')
parser.add_argument('-s', '--hidden-size', default='32')
parser.add_argument('-l', '--loss', default='crossentropy', choices=['mse', 'crossentropy'], help='"mse" or "crossentropy"')
parser.add_argument('-d', '--device')

arguments = parser.parse_args()
print(arguments)

assert arguments.device in ['cpu', 'cuda'], 'Device must be either cpu or cuda'
DEVICE = torch.device(arguments.device if torch.cuda.is_available() else 'cpu')

EPOCHS = int(arguments.epochs)
hidden_size = int(arguments.hidden_size)
NUM_TICKETS = int(arguments.num_tickets)
loss_fn_type = arguments.loss

results['config'] = {
    'epochs': EPOCHS,
    'hidden_size': hidden_size,
}


def get_multiple_models_streams(n_models, train_loader, test_loader, train_set, test_set, hidden_size=32, silent=True):

    if not silent:
        print('Finding base models...')

    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    
    models: list[PrunableModel] = []
    optimizers = []
    
    if loss_fn_type == 'crossentropy': loss_fn = nn.CrossEntropyLoss(reduction='mean')
    elif loss_fn_type == 'mse': loss_fn = nn.MSELoss(reduction='mean')
    else: raise Exception('Loss fn type was neither "mse" nor "crossentropy"')

    timer = pf()
    for i in range(n_models):
        model = construct_mlp([784, hidden_size, 10], flatten_input=True)
        prunable = PrunableModel(model, device=DEVICE)
        optimizer = torch.optim.Adam(params=prunable.parameters(), lr=0.001)
        models.append(prunable)
        optimizers.append(optimizer)

    train_loop_l1_stream(
        models=models,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizers=optimizers,
        n_epochs=EPOCHS
    )

    for i in range(n_models):
        train_loss, train_acc = evaluate_model(models[i], train_set.data.to(torch.float32)/255.0, train_set.targets, loss_fn)
        test_loss, test_acc = evaluate_model(models[i], test_set.data.to(torch.float32)/255.0, test_set.targets, loss_fn)

        models[i].to(device='cpu')
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        # models.append(prunable)

    timer = pf() - timer

    if not silent: print(f'{n_models} models trained in {timer:0.2f} seconds', flush=True)
    return models, train_losses, train_accs, test_losses, test_accs


train_set, test_set = get_mnist_dataset()
train_loader, test_loader = get_loaders(train_set, test_set, batch_size=64, shuffle=False)

models, train_losses, train_accs, test_losses, test_accs = get_multiple_models_streams(
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

results['model-initializations'] = [prunable.retrieve_unpruned_initialization() for prunable in models]
cpuweights = []
for prunable in models:
    cpuweight = {name: param.cpu().clone().detach() for name, param in prunable.named_parameters()}
    cpuweights.append(cpuweight)
results['model-params'] = cpuweights


if not os.path.exists('experiment_data'):
    os.mkdir('experiment_data')

out_path = f'experiment_data/st-subnetworks-e{EPOCHS}-t{NUM_TICKETS}-s{hidden_size}.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(results, f)

print(f'\nSaved to: {out_path}')