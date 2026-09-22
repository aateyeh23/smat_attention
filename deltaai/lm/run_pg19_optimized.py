"""Audit and enable performance kernels before the original audited entrypoint."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy

if os.environ.get('PG19_MEMORY_VARIANT')!='updated_transport':
    raise ValueError('This optimization is enabled only for GDN + SMAT')
expected=json.loads(os.environ['PG19_EXPECTED_SOURCES'])
for name in ('smat_read_tiled','smat_write_hash_tiled','smat_gdn_tiled','pg19_optimized_kernels'):
    module=importlib.import_module(name)
    path=Path(module.__file__).resolve()
    relative=str(path.relative_to('/opt/pg19-rank64'))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==expected[relative],name
    print('SOURCE_AUDIT '+json.dumps(dict(module=name,path=str(path),sha256=digest)),flush=True)
import pg19_optimized_kernels
pg19_optimized_kernels.enable()
print('GDN_KERNEL_OPTIMIZATION tiled_fp32_v1',flush=True)
runpy.run_path('/opt/pg19-rank64/lm/run_pg19_rank64.py',run_name='__main__')
