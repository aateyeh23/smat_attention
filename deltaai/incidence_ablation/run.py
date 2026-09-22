"""Apply an isolated memory ablation before invoking the unmodified frozen trainer."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
from common import ROOT, bootstrap, verify_sources, atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--max-minutes', type=float, default=100)
    args = p.parse_args()
    bootstrap()
    digest = verify_sources()
    validation = json.loads((ROOT/'validated.json').read_text())
    assert validation['passed'] and validation['source_manifest_sha256'] == digest
    task = json.loads((ROOT/'tasks.json').read_text())[args.index]
    folder = Path(task['result']).parent
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if Path(task['result']).exists():
            return
        import joint_recall
        from model import apply_ablation
        original = joint_recall.make_model

        def make_model(*a, **kw):
            model = original(*a, **kw)
            audit = apply_ablation(model, task['variant'], task['topology'])
            audit.update(training_seed=task['seed'], source_manifest_sha256=digest,
                         initialization='fresh, shared backbone and writer initialization by training seed')
            path = folder/'ablation.json'
            if path.exists():
                assert json.loads(path.read_text()) == audit
            else:
                atomic_json(path, audit)
            return model

        joint_recall.make_model = make_model
        sys.argv = [str(Path(joint_recall.__file__)), '--root', task['root'],
                    '--conditions', 'shared', '--families', 'gdn_current', '--ds', '3',
                    '--seeds', str(task['seed']), '--width', '64', '--epochs', '32',
                    '--batch', '256', '--lr', '.003', '--weight-decay', '.1',
                    '--max-minutes', str(args.max_minutes)]
        joint_recall.main()


if __name__ == '__main__':
    main()
