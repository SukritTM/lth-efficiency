import torch
from torch import nn
 
import numpy as np
 
from LTH.datasets import get_mnist_dataset, get_loaders
from LTH.traineval import train_loop, evaluate_model
from LTH.models import construct_mlp
from LTH.models import PrunableModel
 
import os
import pickle
import argparse
from time import perf_counter as pf
 
 
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-file', required=True, help='Path to subnetworks pkl produced by find_subnetworks.py')
parser.add_argument('-n', '--num-experiments', required=True)
parser.add_argument('-d', '--device', required=True)
 
arguments = parser.parse_args()
print(arguments)
 
assert arguments.device in ['cpu', 'cuda'], 'Device must be either cpu or cuda'
DEVICE = torch.device(arguments.device if torch.cuda.is_available() else 'cpu')
 
num_experiments = int(arguments.num_experiments)
 
# ── Load subnetwork data ──────────────────────────────────────────────────────
print(f'Loading subnetwork data from: {arguments.input_file}')
with open(arguments.input_file, 'rb') as f:
    subnetwork_data = pickle.load(f)
 
cfg           = subnetwork_data['config']
EPOCHS        = cfg['epochs']
num_rounds    = cfg['num_rounds']
pruning_ratio = cfg['pruning_ratio']
hidden_size   = cfg['hidden_size']
NUM_TICKETS   = len(subnetwork_data['winning-tickets'])
 
print(f'Loaded config: {cfg}')
print(f'Number of tickets: {NUM_TICKETS}')
 
# Re-apply seeds for reproducibility
tseed = subnetwork_data['seeds']['torch']
nseed = subnetwork_data['seeds']['numpy']
torch.manual_seed(tseed)
np.random.seed(nseed)



print('Reconstructing prunable models from saved tickets...', end='', flush=True)
models: list[PrunableModel] = []
for idx in range(NUM_TICKETS):
    model = construct_mlp([784, hidden_size, 10], flatten_input=True)
 
    # PrunableModel.__init__ calls reinitialize_randomly() then saves that state.
    # We immediately overwrite both below, so the device here is just 'cpu' for setup.
    prunable = PrunableModel(model, mask=subnetwork_data['winning-ticket-masks'][idx], device=DEVICE)
 
    # Restore the full (unpruned) initialization so apply_saved_initialization()
    # and any future retrieve_*() calls behave correctly.
    full_init = subnetwork_data['winning-tickets-full-initializations'][idx]
    # prunable.saved_initialization = {
    #     key: tensor.clone().detach().cpu() for key, tensor in full_init.items()
    # }
 
    # Restore mask (already CPU tensors from the pkl)
    # prunable.mask = {
    #     key: tensor.clone().detach().cpu()
    #     for key, tensor in subnetwork_data['winning-ticket-masks'][idx].items()
    # }
 
    # Set model weights to winning-ticket init with mask applied
    prunable.apply_saved_initialization()
 
    models.append(prunable)
 
print(f'reconstructed {len(models)} models.\n')




train_set, test_set = get_mnist_dataset()
train_loader, _ = get_loaders(train_set, test_set, batch_size=64, shuffle=False)
 
loss_fn = nn.CrossEntropyLoss(reduction='mean')
fc_test_accs = subnetwork_data['FC-test-accs']  # per-ticket baseline accuracy
 
# ── Reinit trials ─────────────────────────────────────────────────────────────
train_losses, train_accs, test_losses, test_accs = [], [], [], []
hits = {}
 
print('Beginning reinit trials...')
for i in range(num_experiments):
    timer = pf()
 
    prunable_idx = np.random.randint(0, len(models))
    prunable = models[prunable_idx]
 
    prunable.to(device=DEVICE)
 
    # Randomize weights while keeping the mask intact, then train
    prunable.reinitialize_randomly()
    new_optimizer = torch.optim.Adam(params=prunable.parameters(), lr=0.001)
    train_loop(prunable, train_loader, loss_fn, new_optimizer, n_epochs=EPOCHS, silent=True)
 
    train_loss, train_acc = evaluate_model(prunable, train_set.data.to(torch.float32)/255.0, train_set.targets, loss_fn)
    test_loss, test_acc   = evaluate_model(prunable, test_set.data.to(torch.float32)/255.0,  test_set.targets,  loss_fn)
 
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
 
    if test_acc - fc_test_accs[prunable_idx] >= 0.005:
        hits[i] = prunable.retrieve_pruned_initialization()
 
    timer = pf() - timer
    prunable.to(device='cpu')
    print(f'Trial {i+1} complete, took {timer:0.2f} seconds', flush=True)
 
# ── Collect and save results ──────────────────────────────────────────────────
results = dict(subnetwork_data)  # carry forward all subnetwork fields
results['reinit-config']        = {'num_experiments': num_experiments}
results['searchexp-train-loss'] = train_losses
results['searchexp-train-acc']  = train_accs
results['searchexp-test-loss']  = test_losses
results['searchexp-test-acc']   = test_accs
results['search-hits']          = hits
 
print('\nExperiment complete.')
from pprint import pprint
print('Saved:')
pprint(list(results.keys()))
 
if not os.path.exists('experiment_data'):
    os.mkdir('experiment_data')
 
out_path = (
    f'experiment_data/search-data-ss'
    f'-e{EPOCHS}-r{num_rounds}-p{pruning_ratio:0.4f}'
    f'-t{NUM_TICKETS}-n{num_experiments}-s{hidden_size}.pkl'
)
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
 
print(f'\nSaved to: {out_path}')