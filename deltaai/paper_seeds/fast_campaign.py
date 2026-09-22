"""Dispatch the separately frozen fast five-seed Log-Linear campaign."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import campaign

OLD_ROOT=campaign.ROOT
ROOT=OLD_ROOT.with_name('paper_fast_20260920')
original_environment=campaign.environment

def environment(task,device):
    # Build imports from the original frozen trainer/backbones, then override
    # only the operator module, runner, and output/dataset root.
    campaign.ROOT=OLD_ROOT
    try:
        env,command=original_environment(task,device)
    finally:
        campaign.ROOT=ROOT
    env['PYTHONPATH']=str(ROOT/'source')+':'+env['PYTHONPATH']
    env['TRITON_CACHE_DIR']=f'/tmp/{env["USER"]}-paper-fast-triton'
    command[3]=str(Path(__file__).with_name('run_joint_fast.py'))
    command[command.index('--root')+1]=str(ROOT/'joint_loglinear')
    return env,command

campaign.ROOT=ROOT
campaign.environment=environment
if __name__=='__main__':
    for rel,expected in json.loads((OLD_ROOT/'source_sha256.json').read_text()).items():
        assert hashlib.sha256((OLD_ROOT/rel).read_bytes()).hexdigest()==expected,rel
    campaign.main()
