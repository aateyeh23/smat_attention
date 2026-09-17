"""Audit loaded sources before validation or pretraining."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
expected=json.loads(os.environ['PG19_EXPECTED_SOURCES'])
modules=('content_addr','smat_read_triton','smat_write_hash_grad','smat_address_reads',
    'smat_gdn_transport','zoo_gdn_transport','zoo_smat_gdn','zoo_smat_mixer',
    'gdn_smat_scale','gdn_smat_transport_scale','mamba_smat_scale','train_pg19_scale')
for name in modules:
    module=importlib.import_module(name)
    path=Path(module.__file__).resolve()
    relative=str(path.relative_to('/opt/pg19-rank64'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[relative],name
    print('SOURCE_AUDIT '+json.dumps(dict(module=name,path=str(path),sha256=expected[relative])),flush=True)
print('WRITE_HASH_BACKEND '+os.environ.get('SMAT_WRITE_HASH_BACKEND','torch'),flush=True)
mode=sys.argv.pop(1)
assert mode in ('validate','train')
script='test_pg19_rank64_gpu.py' if mode=='validate' else 'train_pg19_scale.py'
runpy.run_path('/opt/pg19-rank64/lm/'+script,run_name='__main__')
