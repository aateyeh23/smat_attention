"""Stop a candidate after its checkpoint at the first failed even-epoch gate."""
import json
import os
from pathlib import Path
import torch
from zoo_mqar_interleave import InterleavedTrainer


class BaselineGatedTrainer(InterleavedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.environ.get('GDN_EXPECTED_SOURCES_JSON'):
            from audit_gdn_sources import verify_sources
            verify_sources('training')
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
        variant = json.loads(os.environ.get('GDN_VARIANT_JSON', '{}'))
        self.stop_on_failed_gate = variant.get('stop_on_failed_gate', True)
        if not self.stop_on_failed_gate:
            self.early_stopping_metric = None
        self.max_grad_norm = variant.get('max_grad_norm')
        self.routing_lr_scale = variant.get('routing_lr_scale')
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError('Gradient clipping threshold must be positive')
        if self.routing_lr_scale is not None and self.routing_lr_scale <= 0:
            raise ValueError('Routing learning-rate scale must be positive')

    def optimizer_parameters(self):
        if self.routing_lr_scale is None:
            return self.model.parameters()
        base, routing, names = [], [], []
        for name, parameter in self.model.named_parameters():
            if '.mixer.ca.' in name or '.mixer.kconv.' in name:
                routing.append(parameter);names.append(name)
            else:
                base.append(parameter)
        assert base and routing
        ids = [id(p) for p in base+routing]
        assert len(ids)==len(set(ids))==len(list(self.model.parameters()))
        row = dict(base_lr=self.learning_rate,
                   routing_lr=self.learning_rate*self.routing_lr_scale,
                   routing_parameter_names=names,
                   base_trainable_parameters=sum(p.numel() for p in base if p.requires_grad),
                   routing_trainable_parameters=sum(p.numel() for p in routing if p.requires_grad))
        (Path(os.environ['MQAR_RUN_DIR'])/'optimizer-groups.json').write_text(json.dumps(row,indent=2)+'\n')
        print('OPTIMIZER_GROUPS '+json.dumps({k:v for k,v in row.items() if k!='routing_parameter_names'}),flush=True)
        return [dict(params=base,lr=self.learning_rate,name='base'),
                dict(params=routing,lr=row['routing_lr'],name='routing')]

    def train_epoch(self, epoch_idx):
        if self.max_grad_norm is not None:
            self._gradient_norms = []
            if getattr(self, '_clipped_optimizer', None) is not self.optimizer:
                def clip_before_step(optimizer, args, kwargs):
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm,
                        error_if_nonfinite=True)
                    self._gradient_norms.append(norm.detach())
                self._clip_hook = self.optimizer.register_step_pre_hook(clip_before_step)
                self._clipped_optimizer = self.optimizer
        result = super().train_epoch(epoch_idx)
        if self.max_grad_norm is not None:
            norms = torch.stack(self._gradient_norms).float()
            row = dict(epoch=epoch_idx+1, threshold=self.max_grad_norm,
                       steps=norms.numel(), mean=norms.mean().item(),
                       p95=norms.quantile(.95).item(), maximum=norms.max().item(),
                       clipped_fraction=(norms>self.max_grad_norm).float().mean().item())
            with (Path(os.environ['MQAR_RUN_DIR'])/'gradient-clipping.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print('GRADIENT_CLIP '+json.dumps(row),flush=True)
        return result

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
        decision = ('reject' if self.stop_on_failed_gate else 'below_baseline') if failed else ('pass' if required else 'observe')
        row = dict(epoch=epoch, accuracy=accuracy, baseline=reference,
                   margin=accuracy-reference, required=required, decision=decision)
        folder = Path(os.environ['MQAR_RUN_DIR'])
        with (folder/'gates.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        (folder/'decision.json').write_text(json.dumps(row,indent=2)+'\n')
        print('GATE '+json.dumps(row),flush=True)
        return metrics
