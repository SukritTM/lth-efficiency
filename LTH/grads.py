import torch
from torch import nn

from LTH.models import PrunableModel

class GradRecorder:
    '''
        Records parameter gradients at a fixed, pre-selected set of optimizer steps: the first `early_record_count`
        steps, plus `sample_record_count` steps drawn uniformly without replacement from the whole run.
        Call `on_step(model)` after loss.backward() and before optimizer.step(), on every step. Retrieve with `result()`.
    '''

    def __init__(self, total_steps: int, early_record_count: int = 0, sample_record_count: int = 0, seed: int | None = None, dtype: torch.dtype = torch.bfloat16):
        assert 0 <= early_record_count <= total_steps, 'early_record_count must be between 0 and total_steps'
        assert 0 <= sample_record_count <= total_steps, 'sample_record_count must be between 0 and total_steps'

        self.gen = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else torch.Generator(device='cpu')
        self.seed = self.gen.initial_seed()
        self.dtype = dtype
        self.total_steps = total_steps

        self.sample_indices = torch.randperm(total_steps, generator=self.gen)[:sample_record_count].sort().values
        self.early_indices = torch.arange(early_record_count, dtype=torch.int64)

        # union of both selections; a step picked by both is stored once, with both flags set
        self.steps = torch.cat([self.early_indices, self.sample_indices]).unique(sorted=True)
        self.early = torch.isin(self.steps, self.early_indices)
        self.sampled = torch.isin(self.steps, self.sample_indices)
        self._slot = {step: slot for slot, step in enumerate(self.steps.tolist())}

        self.grads: dict[str, torch.Tensor] | None = None  # allocated on the first recorded step, once shapes are known
        self.internal_counter = 0


    def on_step(self, model: nn.Module):
        slot = self._slot.get(self.internal_counter)
        self.internal_counter += 1
        if slot is None: return

        # unwrap PrunableModel so gradient keys match mask keys ('1.weight', not 'model.1.weight')
        if isinstance(model, PrunableModel): model = model.model

        if self.grads is None:
            n = len(self.steps)
            self.grads = {name: torch.zeros((n, *param.shape), dtype=self.dtype) for name, param in model.named_parameters()}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.grads[name][slot].copy_(param.grad)  # copy_ handles the device move and dtype cast


    def result(self) -> dict:
        return {
            'steps':       self.steps,
            'early':       self.early,
            'sampled':     self.sampled,
            'grads':       self.grads if self.grads is not None else {},
            'total_steps': self.total_steps,
            'steps_seen':  self.internal_counter,
            'seed':        self.seed,
            'dtype':       str(self.dtype),
        }
