"""Stop a candidate after its checkpoint at the first failed even-epoch gate."""
import json
import os
from pathlib import Path
from zoo_mqar_interleave import InterleavedTrainer


class BaselineGatedTrainer(InterleavedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = Path(os.environ['GDN_BASELINE_HISTORY'])
        self.baseline = {r['epoch']: r['metrics']['valid/accuracy']
                         for r in map(json.loads, path.read_text().splitlines())}
        required = set(range(4, self.max_epochs + 1, 2))
        if not required.issubset(self.baseline):
            raise ValueError('Baseline is missing required epoch gates')
        # ResumableTrainer saves the checkpoint before honoring this stop.
        # Disable the unrelated >99% early-stop rule: every required gate counts.
        self.early_stopping_metric = 'gate/failed'
        self.early_stopping_threshold = 0.5

    def test(self, epoch_idx):
        metrics = super().test(epoch_idx)
        epoch = epoch_idx + 1
        required = epoch >= 4 and epoch % 2 == 0
        accuracy = float(metrics['valid/accuracy'])
        reference = float(self.baseline[epoch])
        failed = required and not accuracy > reference
        metrics['gate/failed'] = float(failed)
        metrics['gate/baseline_accuracy'] = reference
        metrics['gate/margin'] = accuracy - reference
        decision = 'reject' if failed else ('pass' if required else 'observe')
        row = dict(epoch=epoch, accuracy=accuracy, baseline=reference,
                   margin=accuracy-reference, required=required, decision=decision)
        folder = Path(os.environ['MQAR_RUN_DIR'])
        with (folder/'gates.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        (folder/'decision.json').write_text(json.dumps(row,indent=2)+'\n')
        print('GATE '+json.dumps(row),flush=True)
        return metrics
