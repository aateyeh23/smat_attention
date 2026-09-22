import os,signal,json
from pathlib import Path
o=json.loads(Path('results/mqar_dispatch/claims/routegrad-32-3/owner.json').read_text())
assert o['job']==os.environ['SLURM_JOB_ID']
p=o['pid']
children=Path(f'/proc/{p}/task/{p}/children').read_text().split()
for child in children:
 cmd=Path(f'/proc/{child}/cmdline').read_bytes()
 if b'zoology.launch' in cmd and b'zoo_gdn_route_grad_configs.py' in cmd:
  os.kill(int(child),signal.SIGTERM);print('Stopped route-gradient trainer',child,flush=True)
