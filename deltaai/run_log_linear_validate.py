"""The GPU correctness gate of modal_log_linear.validate, run on a Slurm GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

LOCAL = Path(__file__).resolve().parent
root = LOCAL / 'results/mqar_log_linear/validation'
root.mkdir(parents=True, exist_ok=True)
with (root / 'gpu.log').open('a') as log:
    child = subprocess.Popen([sys.executable, '-u', str(LOCAL / 'test_log_linear_gpu.py')],
                             cwd=str(LOCAL), env=dict(os.environ),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in child.stdout:
        log.write(line); log.flush(); print(line, end='', flush=True)
    code = child.wait()
if code:
    raise SystemExit(f'gpu validation exit {code}')
sha = hashlib.sha256(Path(os.environ['LL_SOURCE_PATH']).read_bytes()).hexdigest()
(root / 'passed.json').write_text(json.dumps(dict(sha256=sha, passed=True)))
print('VALIDATION PASSED ' + sha, flush=True)
