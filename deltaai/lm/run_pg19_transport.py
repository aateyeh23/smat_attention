"""Verify the actual imports before PG19 GPU validation or training."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
expected=json.loads(os.environ['PG19_EXPECTED_SOURCES'])
modules=('content_addr','smat_read_triton','smat_write_hash_grad','smat_address_reads',
         'smat_gdn_transport','zoo_gdn_transport','zoo_smat_gdn',
         'gdn_smat_scale','gdn_smat_transport_scale','train_pg19_scale')
for name in modules:
    mod=importlib.import_module(name)
    path=Path(mod.__file__).resolve()
    relative=str(path.relative_to('/opt/pg19-updated'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[relative],name
    print('SOURCE_AUDIT '+json.dumps(dict(module=name,path=str(path),sha256=expected[relative])),flush=True)
mode=sys.argv.pop(1)
script='test_pg19_transport_gpu.py' if mode=='validate' else 'train_pg19_scale.py'
assert mode in ('validate','train')
runpy.run_path('/opt/pg19-updated/lm/'+script,run_name='__main__')
