import argparse
import fcntl
import json
import time
from common import ROOT, SEEDS, STATE, atomic_json, bootstrap, make_config, verify_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    seed = SEEDS[args.index]
    bootstrap()
    digest = verify_sources()
    validation = json.loads((ROOT/'validated.json').read_text())
    assert validation['source_manifest_sha256'] == digest and validation['passed']
    folder = ROOT/f'state{STATE}-s{seed}'
    folder.mkdir(exist_ok=True)
    with (folder/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder/'result.json').exists():
            return
        config = make_config(seed, folder)
        config_path = folder/'config.json'
        value = json.loads(config.model_dump_json())
        if config_path.exists():
            assert json.loads(config_path.read_text()) == value
        else:
            atomic_json(config_path, value)
        import torch
        import zoology.train as training
        from zoo_mqar_interleave import InterleavedTrainer
        from tqdm import tqdm
        torch.set_num_threads(2)
        assert torch.cuda.is_available()
        class MatchedTrainer(InterleavedTrainer):
            def train_epoch(self, epoch_idx):
                start = time.monotonic()
                out = super().train_epoch(epoch_idx)
                print(f'TRAIN_EPOCH {epoch_idx+1} seconds={time.monotonic()-start:.2f}', flush=True)
                return out

            def compute_loss(self, inputs, targets):
                loss, preds = super().compute_loss(inputs, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training/evaluation loss')
                return loss, preds

        training.Trainer = MatchedTrainer
        training.tqdm = lambda *a, **kw: tqdm(*a, **dict(kw, disable=True))
        started = time.monotonic()
        training.train(config)
        status = json.loads((folder/'w16-d1.json').read_text())
        timing = folder/'timing.json'
        previous = json.loads(timing.read_text()) if timing.exists() else {'seconds': 0}
        atomic_json(timing, {'seconds': previous['seconds']+time.monotonic()-started})
        if status['complete']:
            assert status['next_epoch'] == 32
            atomic_json(folder/'result.json', dict(seed=seed, state_dim=STATE,
                source_manifest_sha256=digest, **status))


if __name__ == '__main__':
    main()
