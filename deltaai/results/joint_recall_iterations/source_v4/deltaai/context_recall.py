"""Context-key binding, with fixed record count/length and controlled key reuse.

Each shuffled record is (context,key,value); each query is (QUERY,context,key).
This is an explicit-context joint-recall variant, not the paper's block encoding.
All architectures use the unchanged synthetic_memory trainer and model recipes.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from synthetic_memory import run

CONTEXT_BASE, CONTEXT_COUNT = 2, 16
KEY_BASE, KEY_COUNT = 18, 64
VALUE_BASE, VALUES = 82, 16
VOCAB = 98


class ContextRecall:
    def __init__(self, contexts, shared, length):
        if not 0 <= shared <= 1:
            raise ValueError('Shared-key fraction must be between zero and one')
        self.contexts, self.shared, self.length = contexts, shared, length

    def task_tokens(self, task):
        return CONTEXT_COUNT + KEY_COUNT, VALUE_BASE, VOCAB

    def sequence_length(self, task, load, *unused):
        if self.contexts < 2 or self.contexts > CONTEXT_COUNT:
            raise ValueError('Need 2–16 contexts')
        if load % self.contexts or load < 4 or load > KEY_COUNT:
            raise ValueError('Records must divide evenly across contexts, with 4–64 records')
        if 3*load+12 > self.length or self.length == 192:
            raise ValueError('Insufficient sequence length or unsupported common d4 length')
        return self.length

    def generate(self, task, load, batch, rng, *unused):
        length = self.sequence_length(task, load)
        x = np.zeros((batch, length), dtype=np.int64)
        positions = np.tile(np.arange(length-10, length, 3), (batch, 1))
        y = np.empty((batch, 4), dtype=np.int64)
        per_context = load // self.contexts
        shared_count = int(round(per_context*self.shared))
        if not np.isclose(shared_count, per_context*self.shared):
            raise ValueError('Reuse fraction must correspond to a whole number of shared keys')
        for b in range(batch):
            ctx = rng.choice(CONTEXT_COUNT, self.contexts, replace=False)+CONTEXT_BASE
            # Draw the same number of random keys regardless of reuse fraction.
            key_pool = rng.choice(KEY_COUNT, load, replace=False)+KEY_BASE
            keys = key_pool.reshape(self.contexts, per_context).copy()
            keys[:, :shared_count] = key_pool[:shared_count]
            values = rng.integers(VALUES, size=load)+VALUE_BASE
            records = np.column_stack((np.repeat(ctx, per_context), keys.flatten(), values))
            order = rng.permutation(load)
            slots = np.sort(rng.choice((length-12)//3, load, replace=False))*3
            for offset in range(3): x[b, slots+offset] = records[order, offset]
            queries = rng.choice(load, 4, replace=False)
            x[b, positions[b]-2] = 1
            x[b, positions[b]-1] = records[queries, 0]
            x[b, positions[b]] = records[queries, 1]
            y[b] = records[queries, 2]
        return x, positions, y

    @staticmethod
    def table(tokens, positions):
        query_start = int(positions[0])-2
        return [(int(tokens[j]), int(tokens[j+1]), int(tokens[j+2]))
                for j in range(0, query_start-2, 3) if tokens[j]]

    def query_blind_controls(self, task, data):
        key_only, context_only, global_mode = [], [], []
        def mode(values):
            return int(np.bincount(np.asarray(values)-VALUE_BASE, minlength=VALUES).argmax()+VALUE_BASE)
        for tokens, positions in zip(data[0], data[1]):
            records = self.table(tokens, positions)
            key_only.append([mode([v for c,k,v in records if k==tokens[p]]) for p in positions])
            context_only.append([mode([v for c,k,v in records if c==tokens[p-1]]) for p in positions])
            global_mode.append([mode([v for c,k,v in records])]*4)
        return dict(key_only_oracle_accuracy=float((np.asarray(key_only)==data[2]).mean()),
                    context_only_oracle_accuracy=float((np.asarray(context_only)==data[2]).mean()),
                    query_blind_oracle_accuracy=float((np.asarray(global_mode)==data[2]).mean()))

    def binding_control(self, task, data, predictions):
        all_other, context_other, shared_correct = [], [], []
        for tokens, positions, targets, preds in zip(*data, predictions):
            records = self.table(tokens, positions)
            for p, target, pred in zip(positions, targets, preds):
                context, key = int(tokens[p-1]), int(tokens[p])
                others = [v for c,k,v in records if (c,k)!=(context,key)]
                all_other.append(np.mean(np.asarray(others)==pred))
                alternatives = [v for c,k,v in records if k==key and c!=context]
                if alternatives:
                    context_other.append(np.mean(np.asarray(alternatives)==pred))
                    shared_correct.append(float(pred==target))
        accuracy = float((predictions==data[2]).mean())
        return dict(binding_gap_pp=100*(accuracy-float(np.mean(all_other))),
                    context_binding_gap_pp=100*float(np.mean(np.asarray(shared_correct)-context_other)) if context_other else None,
                    wrong_context_accuracy=float(np.mean(context_other)) if context_other else None,
                    shared_query_count=len(context_other))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--records', default='8')
    p.add_argument('--contexts', type=int, default=2)
    p.add_argument('--shared', default='0,1')
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--families', default='mamba2,gdn_current')
    p.add_argument('--ds', default='1,4')
    p.add_argument('--seeds', default='123')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--max-minutes', type=float, default=13)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent/'results/context_recall')
    a = p.parse_args()
    started = time.monotonic()
    for records in map(int, a.records.split(',')):
        for shared in map(float, a.shared.split(',')):
            task = ContextRecall(a.contexts, shared, a.length)
            task.sequence_length('context', records)
            for seed in map(int, a.seeds.split(',')):
                for d in map(int, a.ds.split(',')):
                    for family in a.families.split(','):
                        folder = a.root/f'w{a.width}-r{records}-c{a.contexts}-share{shared:g}-t{a.length}-{family}-d{d}-s{seed}'
                        if (folder/'result.json').exists(): continue
                        remaining = a.max_minutes-(time.monotonic()-started)/60
                        if remaining < 2: return
                        args = argparse.Namespace(task='context', load=records, family=family, d=d,
                            width=a.width, seed=seed, steps=a.steps, batch=32, lr=.01,
                            min_length=a.length, updates=1, hops=0, eval_every=100,
                            eval_examples=1024, test_examples=4096, max_minutes=remaining,
                            contexts=a.contexts, shared=shared, encoding='shuffled explicit-context triples',
                            out=str(folder))
                        run(args, task_api=task)
                        gc.collect()
                        torch.cuda.empty_cache()
                        if not (folder/'result.json').exists(): return
    print('CAMPAIGN_COMPLETE', flush=True)


if __name__ == '__main__': main()
