"""Natural database selection: single predicates and two-field conjunctions.

Predict membership for every database object, without answer feedback. Attribute
value pairs held out of training queries test compositional generalization.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from synthetic_memory import run

PAD, ANY, QUERY = 0, 1, 2
ID_BASE, ID_COUNT = 3, 16
ATTR_BASE = np.array([19, 23, 27])
YES, NO, VOCAB = 31, 32, 33
PAIRS = [(0, 1), (0, 2), (1, 2)]
SEEN = [(a, b) for a in range(4) for b in range(4) if a != b]
HELDOUT = [(a, a) for a in range(4)]


class DatabaseRecall:
    def __init__(self, length=64, loss_mode='unweighted'):
        self.length, self.loss_mode = length, loss_mode

    def task_tokens(self, task):
        return 28, YES, VOCAB

    def recipe_metadata(self):
        return dict(values=2, chance=.5, metric='full-vocabulary membership accuracy; exact sets and balanced accuracy in final diagnostics',
                    schema=['owner', 'color', 'location'], attribute_values=4, queries=2,
                    holdout='equal category indices across distinct attribute domains, in conjunction queries only',
                    primary_split='seen predicate combinations; fresh databases')

    def sequence_length(self, task, load, *unused):
        if not 2 <= load <= ID_COUNT or 4*load + 2*(4+load) > self.length or self.length == 192:
            raise ValueError('Database and two membership queries must fit the common sequence length')
        return self.length

    def training_loss(self, logits, targets, data):
        losses = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), reduction='none').reshape_as(targets)
        if self.loss_mode == 'unweighted': return losses.mean()
        x, positions, _ = data
        load = targets.shape[1]//2
        arity = np.array([[np.sum(x[b, p[q*load]-3:p[q*load]] != ANY) for q in range(2)]
                          for b,p in enumerate(positions)])
        prior = torch.as_tensor(np.repeat(4.**(-arity), load, axis=1), device=targets.device, dtype=torch.float32)
        weights = torch.where(targets==YES, .5/prior, .5/(1-prior))
        return (losses*weights).sum()/weights.sum()

    def generate(self, task, load, batch, rng, *unused, heldout=False):
        length = self.sequence_length(task, load)
        x = np.zeros((batch, length), dtype=np.int64)
        positions = np.empty((batch, 2*load), dtype=np.int64)
        targets = np.empty_like(positions)
        start = length - 2*(4+load)
        combinations = HELDOUT if heldout else SEEN
        for b in range(batch):
            ids = rng.choice(ID_COUNT, load, replace=False)+ID_BASE
            attrs = rng.integers(4, size=(load, 3))
            order = rng.permutation(load)
            slots = np.sort(rng.choice(start//4, load, replace=False))*4
            rows = np.column_stack((ids, attrs+ATTR_BASE))[order]
            for off in range(4): x[b, slots+off] = rows[:, off]
            # One singleton and one conjunction, in random order.
            for qi, arity in enumerate(rng.permutation([1, 2])):
                predicate = np.full(3, ANY, dtype=np.int64)
                if arity == 1:
                    fields = [int(rng.integers(3))]
                    vals = [int(rng.integers(4))]
                else:
                    fields = PAIRS[int(rng.integers(3))]
                    # Same random draw count between seen/heldout evaluations.
                    vals = combinations[min(int(rng.random()*len(combinations)), len(combinations)-1)]
                predicate[list(fields)] = ATTR_BASE[list(fields)]+vals
                qstart = start + qi*(4+load)
                x[b, qstart:qstart+4] = np.r_[QUERY, predicate]
                candidates = rng.permutation(load)
                pos = np.arange(qstart+4, qstart+4+load)
                x[b, pos] = ids[candidates]
                match = (attrs[candidates][:, fields] == np.asarray(vals)).all(axis=1)
                positions[b, qi*load:(qi+1)*load] = pos
                targets[b, qi*load:(qi+1)*load] = np.where(match, YES, NO)
        return x, positions, targets

    @staticmethod
    def metrics(data, predictions):
        x, positions, targets = data
        load = targets.shape[1]//2
        target = targets.reshape(-1, load)
        pred = predictions.reshape(-1, load)
        arity = np.array([np.sum(x[b, p[qi*load]-3:p[qi*load]] != ANY)
                          for b,p in enumerate(positions) for qi in range(2)])
        result = {}
        for name, select in [('all', np.ones(len(target), dtype=bool)), ('single', arity==1), ('conjunction', arity==2)]:
            t, p = target[select], pred[select]
            positive = t == YES
            correct = p == t
            tpr = float(correct[positive].mean())
            tnr = float(correct[~positive].mean())
            exact = correct.all(axis=1)
            nonempty = positive.any(axis=1)
            tp = int(((p==YES)&positive).sum())
            fp = int(((p==YES)&~positive).sum())
            fn = int(((p!=YES)&positive).sum())
            result.update({f'{name}_balanced_accuracy':(tpr+tnr)/2,
                           f'{name}_exact_set_accuracy':float(exact.mean()),
                           f'{name}_nonempty_exact_set_accuracy':float(exact[nonempty].mean()),
                           f'{name}_positive_recall':tpr,
                           f'{name}_negative_recall':tnr,
                           f'{name}_micro_f1':2*tp/max(1,2*tp+fp+fn),
                           f'{name}_positive_fraction':float(positive.mean()),
                           f'{name}_always_no_accuracy':float((~positive).mean()),
                           f'{name}_always_no_exact_set_accuracy':float((~nonempty).mean())})
        return result

    def query_blind_controls(self, task, data):
        return dict(always_no_accuracy=float((data[2]==NO).mean()))

    def binding_control(self, task, data, predictions):
        metrics = self.metrics(data, predictions)
        # Required completion marker used by the existing trainer. Balanced
        # discrimination is zero for constant yes/no predictions.
        return dict(metrics, binding_gap_pp=200*(metrics['all_balanced_accuracy']-.5))

    def extra_evaluation(self, model, args, evaluate):
        data = self.generate('database', args.load, args.test_examples,
            np.random.default_rng(np.random.SeedSequence([args.seed, 30001, args.load])), heldout=True)
        metrics, predictions = evaluate(model, data, args.batch, return_predictions=True)
        np.savez_compressed(Path(args.out)/'heldout_predictions.npz', predictions=predictions, targets=data[2], positions=data[1])
        return dict(heldout_sha256=hashlib.sha256(b''.join(a.tobytes() for a in data)).hexdigest(),
                    **{f'heldout_{k}':v for k,v in dict(metrics, **self.metrics(data, predictions)).items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--records', default='4')
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--families', default='mamba2,gdn_current')
    p.add_argument('--ds', default='1,3')
    p.add_argument('--seeds', default='123')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--loss', choices=['unweighted', 'balanced'], default='unweighted')
    p.add_argument('--max-minutes', type=float, default=13)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent/'results/database_recall')
    a = p.parse_args(); started = time.monotonic(); task = DatabaseRecall(a.length, a.loss)
    for records in map(int, a.records.split(',')):
        task.sequence_length('database', records)
        for seed in map(int, a.seeds.split(',')):
            for d in map(int, a.ds.split(',')):
                for family in a.families.split(','):
                    folder = a.root/f'w{a.width}-r{records}-t{a.length}-{family}-d{d}-s{seed}'
                    if a.loss == 'balanced': folder = folder.with_name(folder.name+'-balanced')
                    if (folder/'result.json').exists(): continue
                    remaining = a.max_minutes-(time.monotonic()-started)/60
                    if remaining < 2: return
                    args = argparse.Namespace(task='database', load=records, family=family, d=d,
                        width=a.width, seed=seed, steps=a.steps, batch=32, lr=.01,
                        min_length=a.length, updates=1, hops=0, eval_every=100,
                        eval_examples=1024, test_examples=4096, max_minutes=remaining,
                        loss_mode=a.loss, encoding='object owner color location; wildcard predicates; all-object membership', out=str(folder))
                    run(args, task_api=task)
                    gc.collect(); torch.cuda.empty_cache()
                    if not (folder/'result.json').exists(): return
    print('CAMPAIGN_COMPLETE', flush=True)


if __name__ == '__main__': main()
