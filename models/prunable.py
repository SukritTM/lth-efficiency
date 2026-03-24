import torch
import torch.nn as nn
import numpy as np


class PrunableModel(nn.Module):
    def __init__(self, model, mask=None, device=torch.device('cpu')):
        super().__init__()
        self.model = model
        self.mask = mask
        self.reinitialize_randomly()
        self.device = device

        self.model.to(device=self.device)
        # save the initialization
        self.saved_initialization = dict()
        for key in dict(model.named_parameters()).keys():
            self.saved_initialization[key] = torch.tensor(dict(model.named_parameters())[key].data.clone().detach().numpy()).to(device=self.device)

        self._apply_mask()
    
    def _apply_mask(self):
        if self.mask is None: return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = param * self.mask[name] 
    
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

    def _reinitialize_randomly_recurse(self, obj: nn.Module):
        for child in obj.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()
            self._reinitialize_randomly_recurse(child)
    
    def find_mask(self, remove_fraction: float,):
        '''
        Takes a pruned model as a (model, mask) pair and removes a specified fraction of weights, returning a new mask corresponding to the 
        new pruned model
        '''
        if self.mask == None:
            # return
            self.mask = {k: np.ones_like(v.detach().cpu().numpy(), dtype=np.float32) for k, v in self.model.named_parameters()}
        elif type(list(self.mask.values())[0]) == torch.Tensor:
            self.mask = {k: v.detach().cpu().numpy() for k, v in self.mask.items()}
        
        num_unpruned_weights = np.concat([v.flatten() for k, v in self.mask.items()]).sum()
        num_weights_to_prune = int(remove_fraction*num_unpruned_weights)
        
        unpruned_weights = np.concat([np.abs(v.detach().cpu().numpy())[self.mask[k] == 1] for k, v in self.model.named_parameters()]).flatten()
        unpruned_weights.sort()
        assert unpruned_weights.shape[0] == num_unpruned_weights # sanity check
        
        thres = unpruned_weights[num_weights_to_prune]

        prev_masked_out_params = {k: v.detach().cpu().numpy()*self.mask[k] for k, v in self.model.named_parameters()}
        new_mask = {k: torch.tensor((np.abs(v) > thres).astype(np.float32)).to(self.device) for k, v in prev_masked_out_params.items()}

        self.mask = new_mask
        self._apply_mask()