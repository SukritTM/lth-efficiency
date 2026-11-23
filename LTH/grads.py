import torch
import numpy as np

class GradRecorder:
    def __init__(self, total_steps: int, early_record_count: int, sample_record_count: int, seed: int | None = None, dtype: torch.dtype = torch.float16):
        self.gen = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else torch.Generator(device='cpu')
        self.dtype = dtype
        self.sample_indices = torch.randint(low=0, high=total_steps, size=(sample_record_count,), generator=self.gen).sort().indices
        self.early_indices = torch.range(start=0, stop=early_record_count, step=1, dtype=torch.int32)

        self.internal_counter = 0


    def on_step(self, model: torch.Module):

        if (self.internal_counter in self.sample_indices) or (self.internal_counter in self.early_indices):
            self.internal_counter += 1 

            grads = []
            for name, param in model.named_parameters():
                grads.append((name, param.grad.cpu()))
            
            return grads