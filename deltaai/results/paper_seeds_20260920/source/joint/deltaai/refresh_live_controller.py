"""Restart only the dispatcher controller; leave its current trainer untouched."""
import json,os,signal,time
from pathlib import Path
p=Path('results/mqar_dispatch/controller.json')
pid=json.loads(p.read_text())['pid']
command=Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'mqar_dispatch.py' in command and b'controller' in command, command
os.kill(pid,signal.SIGTERM)
for _ in range(30):
    if not Path(f'/proc/{pid}').exists(): break
    time.sleep(.1)
os.execv('/bin/bash',['bash','mqar_dispatch_env.sh','controller'])
