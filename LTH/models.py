import torch
import torch.nn as nn
import numpy as np


class PrunableModel(nn.Module):
    def __init__(self, model, mask=None, device=torch.device('cpu'), prune_exclude=None):
        '''
        `prune_exclude` protects parameters from find_mask. It may be:
          - a str: a full parameter name ('3.weight') or a parameter type ('weight', 'bias') matching that type in every layer
          - a list/set of such strs, combined
          - a dict mapping parameter names to boolean tensors of the parameter's shape, True marking protected entries
        Protected entries always keep mask value 1.
        '''
        super().__init__()
        self.model = model
        self.mask = mask
        self.reinitialize_randomly()
        self.device = device

        self.model.to(device=self.device)
        if mask is not None: self.mask = {k: 
                                          torch.tensor(v).to(device=self.device) if type(v) is not torch.Tensor else v.clone().detach().to(device=self.device) 
                                          for k, v in mask.items()}

        self.prunable = self._build_prunable(prune_exclude)
        if self.mask is not None:
            for name, prunable in self.prunable.items():
                if (~prunable & (self.mask[name] == 0)).any():
                    raise ValueError(f'mask prunes excluded entries of {name!r}; excluded entries must have mask value 1')

        # save the initialization
        self.saved_initialization = dict()
        for key in dict(model.named_parameters()).keys():
            self.saved_initialization[key] = torch.tensor(dict(model.named_parameters())[key].data.clone().detach().cpu().numpy()).to(device=self.device)

        self._apply_mask()

    def change_device(self, device):
        self.model.to(device=device)

        if self.mask is not None: self.mask = {k: 
                                               torch.tensor(v).to(device=device) if type(v) is not torch.Tensor else v.clone().detach().to(device=device) 
                                               for k, v in self.mask.items()}

        for key, value in self.saved_initialization.items():
            self.saved_initialization[key] = value.to(device)

        self.prunable = {k: v.to(device) for k, v in self.prunable.items()}

        self.device = device
    
    def _build_prunable(self, prune_exclude):
        params = dict(self.model.named_parameters())
        prunable = {k: torch.ones_like(v, dtype=torch.bool, device=self.device) for k, v in params.items()}
        if prune_exclude is None:
            return prunable

        if isinstance(prune_exclude, dict):
            for name, excluded in prune_exclude.items():
                if name not in params:
                    raise ValueError(f'prune_exclude names unknown parameter {name!r}; parameters: {sorted(params)}')
                excluded = torch.as_tensor(excluded, dtype=torch.bool)
                if excluded.shape != params[name].shape:
                    raise ValueError(f'prune_exclude[{name!r}] has shape {tuple(excluded.shape)}, parameter has shape {tuple(params[name].shape)}')
                prunable[name] = ~excluded.to(self.device)
            return prunable

        # check str first: a str is iterable and would otherwise be split into characters
        if isinstance(prune_exclude, str):
            prune_exclude = [prune_exclude]

        for spec in prune_exclude:
            if spec in params:
                matched = [spec]
            else:
                matched = [name for name in params if name.rsplit('.', 1)[-1] == spec]
            if not matched:
                types = sorted({name.rsplit('.', 1)[-1] for name in params})
                raise ValueError(f'prune_exclude entry {spec!r} matches no parameter; names: {sorted(params)}, types: {types}')
            for name in matched:
                prunable[name] = torch.zeros_like(prunable[name])

        return prunable

    def _apply_mask(self):
        if self.mask is None: return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # print(f'{param.device=}')
                # print(f'{self.mask[name]=}')
                try:
                    param.data = param * self.mask[name] 
                except RuntimeError as e:
                    print(f'{name=}')
                    print(f'{param.device=}')
                    print(f'{self.mask[name].device=}')
                    raise e
                
    def retrieve_pruned_initialization(self):
        initialization = {key: self.saved_initialization[key].clone().detach() for key in self.saved_initialization.keys()}
        if self.mask is None: return initialization
        with torch.no_grad():
            for name in initialization.keys():
                param = initialization[name]
                initialization[name] = (param * self.mask[name]).cpu() 
        
        return initialization
    
    def retrieve_unpruned_initialization(self):
        initialization = {key: self.saved_initialization[key].clone().detach() for key in self.saved_initialization.keys()}
        if self.mask is None: return initialization
        with torch.no_grad():
            for name in initialization.keys():
                param = initialization[name]
                initialization[name] = (param).cpu() 
        
        return initialization

    def forward(self, x):
        self._apply_mask()
        return self.model(x)
    
    def apply_saved_initialization(self):
        for name, param in self.model.named_parameters():
            param.data = self.saved_initialization[name].clone().detach()
            # param.copy_(self.saved_initialization[name].clone().detach())
        self._apply_mask()
    
    def reinitialize_randomly(self):
        self._reinitialize_randomly_recurse(self.model)
        self._apply_mask()
        if hasattr(self, 'saved_initialization'):
            self.saved_initialization = {
                name: param.detach().clone().to(self.device)
                for name, param in self.model.named_parameters()
            }

    def _reinitialize_randomly_recurse(self, obj: nn.Module):
        for child in obj.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()
            self._reinitialize_randomly_recurse(child)
    
    def find_mask(self, remove_fraction: float,):
        '''
        Takes a pruned model as a (model, mask) pair and removes a specified fraction of weights, returning a new mask corresponding to the
        new pruned model. 
        `remove_fraction` is a fraction of the currently-alive *prunable* entries; entries excluded via `prune_exclude` are neither
        counted nor removed.
        '''
        if self.mask == None:
            # return
            self.mask = {k: np.ones_like(v.detach().cpu().numpy(), dtype=np.float32) for k, v in self.model.named_parameters()}
        elif type(list(self.mask.values())[0]) == torch.Tensor:
            self.mask = {k: v.detach().cpu().numpy() for k, v in self.mask.items()}
        
        prunable = {k: v.detach().cpu().numpy() for k, v in self.prunable.items()}
        candidates = {k: (self.mask[k] == 1) & prunable[k] for k in self.mask.keys()}

        num_unpruned_weights = int(sum(c.sum() for c in candidates.values()))
        num_weights_to_prune = int(remove_fraction*num_unpruned_weights)
        
        if num_unpruned_weights == 0:
            self.mask = {k: torch.tensor(v).to(self.device) for k, v in self.mask.items()}
            self._apply_mask()
            return
        
        unpruned_weights = np.concat([np.abs(v.detach().cpu().numpy())[candidates[k]] for k, v in self.model.named_parameters()]).flatten()
        unpruned_weights.sort()
        assert unpruned_weights.shape[0] == num_unpruned_weights # sanity check
        
        thres = unpruned_weights[min(num_weights_to_prune, num_unpruned_weights - 1)]

        prev_masked_out_params = {k: v.detach().cpu().numpy()*self.mask[k] for k, v in self.model.named_parameters()}
        new_mask = {k: torch.tensor(((np.abs(v) > thres) | ~prunable[k]).astype(np.float32)).to(self.device) for k, v in prev_masked_out_params.items()}

        self.mask = new_mask
        self._apply_mask()

    def sparsity_stats(self):
        n_total, n_excluded, n_alive, n_prunable_alive = 0, 0, 0, 0
        for name, prunable in self.prunable.items():
            prunable = prunable.detach().cpu()
            if self.mask is None:
                alive = torch.ones_like(prunable)
            else:
                alive = torch.as_tensor(self.mask[name]).detach().cpu() == 1

            n_total += prunable.numel()
            n_excluded += int((~prunable).sum())
            n_alive += int(alive.sum())
            n_prunable_alive += int((alive & prunable).sum())

        n_prunable = n_total - n_excluded
        return {
            'n_total': n_total,
            'n_excluded': n_excluded,
            'n_alive': n_alive,
            'n_prunable_alive': n_prunable_alive,
            'overall_remaining': n_alive / n_total,
            'prunable_remaining': n_prunable_alive / n_prunable if n_prunable > 0 else float('nan'),
        }

def pruning_ratio_for_target(target_remaining: float, n_total: int, n_excluded: int, num_rounds: int) -> float:
    '''
    Per-round `remove_fraction` for find_mask that leaves `target_remaining` of all `n_total` parameters alive after `num_rounds`
    rounds, when `n_excluded` of them are excluded from pruning:
        (1 - p)^r = (f*N - E) / (N - E)
    Ignores find_mask's integer truncation and tie handling.
    '''
    if not 0 <= target_remaining <= 1:
        raise ValueError(f'target_remaining must be in [0, 1], got {target_remaining}')
    if num_rounds < 1:
        raise ValueError(f'num_rounds must be at least 1, got {num_rounds}')
    if n_excluded >= n_total:
        raise ValueError(f'all {n_total} parameters are excluded; nothing can be pruned')
    if target_remaining * n_total < n_excluded:
        raise ValueError(f'target of {target_remaining * n_total:.1f} remaining parameters is below the {n_excluded} excluded parameters')

    prunable_remaining = (target_remaining * n_total - n_excluded) / (n_total - n_excluded)
    return 1 - prunable_remaining ** (1 / num_rounds)

def exclude_tag(prune_exclude) -> str:
    '''Filename suffix for a string exclusion spec: '' when nothing is excluded, else e.g. '-xbias' or '-xbias-3.weight'.'''
    if not prune_exclude:
        return ''
    if isinstance(prune_exclude, str):
        prune_exclude = [prune_exclude]
    return '-x' + '-'.join(prune_exclude)

def construct_mlp(layer_config: list[int], flatten_input = False) -> nn.Sequential:
    layers = [nn.Flatten()] if flatten_input else []
    for i in range(len(layer_config) - 1):
        in_features = layer_config[i]
        out_features = layer_config[i+1]
        layers.append(nn.Linear(in_features, out_features))

        if i < len(layer_config) - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)

