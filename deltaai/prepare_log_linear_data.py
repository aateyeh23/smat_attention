"""Build the MQAR mixture cache once, before the six arms start.

On Modal each arm ran in its own container with its own copy of zoology_cache.
On SCF the six arms share one filesystem, so six simultaneous first epochs would
generate and torch.save the same cache files at the same time. This generates
them once; the arms then only read.
"""
import os
from zoo_reference_configs import data
from zoology.data.utils import prepare_data

os.makedirs(data.cache_dir, exist_ok=True)
train, test = prepare_data(data)
print(f'train batches {len(train)}  test batches {len(test)}', flush=True)
for name, loader in (('train', train), ('test', test)):
    x, y, s = next(iter(loader))
    print(f'{name}: inputs {tuple(x.shape)} targets {tuple(y.shape)}', flush=True)
print('DATA CACHE READY ' + data.cache_dir, flush=True)
