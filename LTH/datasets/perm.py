import torch

class PermutePixels(torch.nn.Module):

    def __init__(self, seed=None):
        super().__init__()
        self.gen = torch.Generator()
        self.gen.seed() if seed is None else self.gen.manual_seed(seed)
        
        self.perm = None
    
    def forward(self, img: torch.Tensor):
        assert len(img.shape) == 3
        assert img.shape[0] == 1
        
        old_shape = img.shape

        flat_view = img.flatten(start_dim=1)
        
        if self.perm is None: self.perm = torch.randperm(flat_view.shape[-1], generator=self.gen) 

        permuted = flat_view[:, self.perm]
        unflattened = permuted.view(old_shape)

        return unflattened
