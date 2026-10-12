import torch
import numpy as np

from LTH.models import PrunableModel, construct_mlp, pruning_ratio_for_target, exclude_tag

torch.manual_seed(0)

LAYERS = [6, 4, 3]  # parameters: 1.weight (4, 6), 1.bias (4,), 3.weight (3, 4), 3.bias (3,)


def make_prunable(**kwargs):
    return PrunableModel(construct_mlp(LAYERS, flatten_input=True), **kwargs)


def mask_np(prunable):
    return {k: torch.as_tensor(v).detach().cpu().numpy() for k, v in prunable.mask.items()}


def legacy_find_mask(params, mask, remove_fraction):
    '''find_mask's selection logic as it was before prune_exclude existed, on numpy arrays.'''
    num_unpruned_weights = np.concat([v.flatten() for v in mask.values()]).sum()
    num_weights_to_prune = int(remove_fraction*num_unpruned_weights)
    unpruned_weights = np.concat([np.abs(params[k])[mask[k] == 1] for k in params]).flatten()
    unpruned_weights.sort()
    thres = unpruned_weights[num_weights_to_prune]
    return {k: (np.abs(params[k]*mask[k]) > thres).astype(np.float32) for k in params}


def check_rounds(prunable, remove_fraction, num_rounds):
    '''Run find_mask rounds, checking the exclusion invariant, monotonicity, removal counts and sparsity_stats.'''
    excluded = {k: ~v.cpu().numpy() for k, v in prunable.prunable.items()}
    original = {k: v.detach().cpu().numpy().copy() for k, v in prunable.model.named_parameters()}
    prev = {k: np.ones_like(v) for k, v in original.items()}

    for _ in range(num_rounds):
        prev_pool = sum(int(((prev[k] == 1) & ~excluded[k]).sum()) for k in prev)
        prunable.find_mask(remove_fraction)
        new = mask_np(prunable)
        params = {k: v.detach().cpu().numpy() for k, v in prunable.model.named_parameters()}

        for k in new:
            assert (new[k][excluded[k]] == 1).all(), f'{k}: excluded entry pruned'
            assert np.array_equal(params[k][excluded[k]], original[k][excluded[k]]), f'{k}: excluded value changed'
            assert (new[k] <= prev[k]).all(), f'{k}: pruned entry came back'

        removed = prev_pool - sum(int(((new[k] == 1) & ~excluded[k]).sum()) for k in new)
        # strict '>' against a pool value removes one more than requested (no ties with random weights), capped at the pool
        expected = min(int(remove_fraction*prev_pool) + 1, prev_pool)
        assert removed == expected, f'removed {removed}, expected {expected} from a pool of {prev_pool}'

        stats = prunable.sparsity_stats()
        assert stats['n_total'] == sum(v.size for v in new.values())
        assert stats['n_excluded'] == sum(int(v.sum()) for v in excluded.values())
        assert stats['n_alive'] == sum(int((v == 1).sum()) for v in new.values())
        assert stats['n_prunable_alive'] == prev_pool - removed

        prev = new


def expect_value_error(fn, label):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f'{label}: expected ValueError')


# no exclusion: masks identical to the old find_mask, round by round
prunable = make_prunable()
params = {k: v.detach().cpu().numpy().copy() for k, v in prunable.model.named_parameters()}
legacy = {k: np.ones_like(v) for k, v in params.items()}
for _ in range(4):
    prunable.find_mask(0.3)
    legacy = legacy_find_mask(params, legacy, 0.3)
    new = mask_np(prunable)
    assert all(np.array_equal(new[k], legacy[k]) for k in new), 'mask differs from old find_mask'
check_rounds(make_prunable(), 0.3, 4)
print('ok: no exclusion matches old find_mask')

# exclude a type in every layer
prunable = make_prunable(prune_exclude='bias')
assert not prunable.prunable['1.bias'].any() and not prunable.prunable['3.bias'].any()
assert prunable.prunable['1.weight'].all() and prunable.prunable['3.weight'].all()
check_rounds(prunable, 0.3, 4)
print("ok: 'bias'")

# exclude weights: only the 7 biases are prunable, so the pool runs out and the empty-pool path is taken
prunable = make_prunable(prune_exclude='weight')
check_rounds(prunable, 0.5, 6)
assert prunable.sparsity_stats()['n_prunable_alive'] == 0
assert all(isinstance(v, torch.Tensor) for v in prunable.mask.values()), 'mask left as numpy after empty pool'
prunable(torch.randn(2, 6))  # forward pass still works
print("ok: 'weight' (pool exhausted)")

# union of a type and a full name
prunable = make_prunable(prune_exclude=['bias', '3.weight'])
assert prunable.prunable['1.weight'].all()
assert not any(prunable.prunable[k].any() for k in ['1.bias', '3.bias', '3.weight'])
check_rounds(prunable, 0.3, 4)
print("ok: ['bias', '3.weight']")

# per-element dict form
protected = torch.zeros(4, 6, dtype=torch.bool)
protected[0, :] = True
protected[2, 3] = True
prunable = make_prunable(prune_exclude={'1.weight': protected})
assert torch.equal(prunable.prunable['1.weight'], ~protected)
assert all(prunable.prunable[k].all() for k in ['1.bias', '3.weight', '3.bias'])
check_rounds(prunable, 0.4, 5)
print('ok: dict form')

# invalid specs
expect_value_error(lambda: make_prunable(prune_exclude='bais'), "'bais'")
expect_value_error(lambda: make_prunable(prune_exclude=['bias', 'nope']), "['bias', 'nope']")
expect_value_error(lambda: make_prunable(prune_exclude={'9.weight': torch.zeros(1, dtype=torch.bool)}), 'unknown dict name')
expect_value_error(lambda: make_prunable(prune_exclude={'1.weight': torch.zeros(6, 4, dtype=torch.bool)}), 'wrong dict shape')
print('ok: invalid specs raise')

# incoming mask that prunes an excluded entry
mask = {k: torch.ones_like(v) for k, v in construct_mlp(LAYERS, flatten_input=True).named_parameters()}
make_prunable(mask=mask, prune_exclude='bias')  # consistent mask is accepted
mask['1.bias'][1] = 0
expect_value_error(lambda: make_prunable(mask=mask, prune_exclude='bias'), 'mask violating exclusion')
make_prunable(mask=mask)  # same mask is fine without exclusion
print('ok: mask invariant check')

# change_device moves prunable with the mask
if torch.cuda.is_available():
    prunable = make_prunable(prune_exclude='bias')
    prunable.find_mask(0.3)
    prunable.change_device(torch.device('cuda'))
    assert all(v.device.type == 'cuda' for v in prunable.prunable.values())
    prunable.find_mask(0.3)
    print('ok: change_device (cuda)')
else:
    print('skipped: change_device (no cuda)')

# pruning_ratio_for_target
for f, N, E, r in [(0.1, 25450, 0, 10), (0.1, 25450, 42, 10), (0.4, 25450, 330, 10), (0.05, 1000, 20, 3)]:
    p = pruning_ratio_for_target(f, N, E, r)
    assert abs((E + (N - E)*(1 - p)**r)/N - f) < 1e-12, (f, N, E, r)
assert round(pruning_ratio_for_target(0.1, 25450, 0, 10), 4) == 0.2057  # job-ticket-sizes-fixed.sh, h=32
expect_value_error(lambda: pruning_ratio_for_target(0.01, 1000, 20, 10), 'target below excluded count')
expect_value_error(lambda: pruning_ratio_for_target(0.5, 100, 100, 10), 'everything excluded')
print('ok: pruning_ratio_for_target')

# exclude_tag
assert exclude_tag(None) == '' and exclude_tag([]) == ''
assert exclude_tag('bias') == '-xbias'
assert exclude_tag(['bias', '3.weight']) == '-xbias-3.weight'
print('ok: exclude_tag')

print('\nall checks passed')
