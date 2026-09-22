"""Fresh synthetic state and variable tracking; labels only at query identifiers.

Each record is two tokens (identifier, state). Records are randomly interleaved
and placed in random pair slots, including padding gaps. There is no alignment
with the SMat boundary. Query answers never appear as teacher-forced inputs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

PAD, QUERY = 0, 1
ID_BASE = 2
ID_COUNT = 256
VALUE_BASE = ID_BASE + ID_COUNT
VALUES = 16
VOCAB = VALUE_BASE + VALUES


def task_tokens(task):
    count = ID_COUNT if task == 'state' else 512
    value_base = ID_BASE + count
    return count, value_base, value_base + VALUES


def sequence_length(task, load, updates=3, hops=2, min_length=256):
    records = load * (updates if task == 'state' else hops + 1)
    queries = 4 if task == 'state' else 1
    length = max(min_length, math.ceil((2 * records + 2 * queries + 32) / 64) * 64)
    # The existing d=4 model cannot instantiate its full geometry at T=192.
    # Apply this padding rule to every architecture, without changing d.
    return 256 if length == 192 else length


def generate(task, load, batch, rng, updates=3, hops=2, min_length=256):
    if task == 'variable':
        return generate_variable(load, batch, rng, hops, min_length)
    length = sequence_length(task, load, updates, hops, min_length)
    x = np.zeros((batch, length), dtype=np.int64)
    y = np.empty((batch, 4), dtype=np.int64)
    positions = np.tile(np.arange(length - 7, length, 2), (batch, 1))
    for b in range(batch):
        ids = rng.choice(ID_COUNT, load, replace=False) + ID_BASE
        owners = np.repeat(np.arange(load), updates)
        rng.shuffle(owners)
        states = rng.integers(VALUES, size=len(owners)) + VALUE_BASE
        latest = np.empty(load, dtype=np.int64)
        slots = np.sort(rng.choice((length - 8) // 2, len(owners), replace=False)) * 2
        x[b, slots] = ids[owners]
        x[b, slots + 1] = states
        for owner, state in zip(owners, states):
            latest[owner] = state
        queries = rng.choice(load, 4, replace=False)
        x[b, positions[b] - 1] = QUERY
        x[b, positions[b]] = ids[queries]
        y[b] = latest[queries]
    return x, positions, y


def generate_variable(load, batch, rng, hops=2, min_length=256):
    # H counts variable-to-variable edges; a terminal assignment follows them.
    # Assignments are declarative links: their sequence order has no semantics.
    count, value_base, _ = task_tokens('variable')
    if load * (hops + 1) > count:
        raise ValueError('Not enough unique identifiers for disjoint chains')
    length = sequence_length('variable', load, hops=hops, min_length=min_length)
    x = np.zeros((batch, length), dtype=np.int64)
    positions = np.full((batch, 1), length - 1, dtype=np.int64)
    y = np.empty((batch, 1), dtype=np.int64)
    for b in range(batch):
        chains = rng.choice(count, load*(hops+1), replace=False).reshape(load, hops+1) + ID_BASE
        values = rng.integers(VALUES, size=load) + value_base
        lhs = chains.flatten()
        rhs = np.concatenate((chains[:, 1:], values[:, None]), axis=1).flatten()
        order = rng.permutation(len(lhs))
        slots = np.sort(rng.choice((length-2)//2, len(lhs), replace=False))*2
        x[b, slots], x[b, slots+1] = lhs[order], rhs[order]
        query = rng.integers(load)
        x[b, -2:] = [QUERY, chains[query, 0]]
        y[b, 0] = values[query]
    return x, positions, y


def make_model(family, width, d, length, vocab=VOCAB):
    lengths = tuple(length) if isinstance(length, (tuple, list)) else (length,)
    import zoology.mixers.mamba2 as zm
    from zoo_smat_mixer import SmatMamba2Block
    zm.Mamba2Block = SmatMamba2Block
    from zoology.config import ModelConfig
    from zoology.model import LanguageModel
    if family == 'mamba2':
        name = 'zoo_mamba_four_reads.MqarMambaFourReads'
        kwargs = dict(d=d, prebuild_lengths=lengths)
    else:
        name = 'zoo_smat_gdn.SmatGDNReset'
        kwargs = dict(d=d, headdim=16, n_heads=2, expand_v=1, reset=True,
                      lam_bias=-2.197224577, anneal_steps=1000, balance_coef=.01,
                      hash_codim=1, memory_update='delta', memory_key_mode='tied_causal',
                      memory_read_k=4 if d >= 2 else None, prebuild_lengths=lengths)
        if family == 'gdn_current' and d >= 2:
            name = 'zoo_gdn_transport.SmatGDNTransport'
            kwargs.update(write_hash_neighbor_grad=True, transport_scalar_decay=True,
                          detach_write_hash_input=True, memory_plant=False,
                          memory_incidence_rescale=False)
    config = ModelConfig(block_type='Mamba2Block', d_model=width, n_layers=2,
        sequence_mixer=dict(name=name, kwargs=kwargs), max_position_embeddings=0,
        vocab_size=vocab, name=f'{family}-d{d}', embed_dropout=.1)
    return LanguageModel(config).cuda()


def query_logits(model, arrays):
    x, positions, targets = [torch.from_numpy(a).cuda() for a in arrays]
    # Same tied vocabulary head as MQAR, evaluated only at supervised positions.
    hidden = model(x, return_embeddings=True)
    logits = model.lm_head(hidden[torch.arange(x.shape[0], device='cuda')[:, None], positions])
    return logits, targets


def auxiliary(model):
    return sum(m.get_auxiliary_loss() for m in model.modules()
               if hasattr(m, 'get_auxiliary_loss'))


@torch.no_grad()
def evaluate(model, data, batch, return_predictions=False):
    model.eval()
    correct = exact = count = examples = 0
    loss = 0.
    predictions = []
    for start in range(0, len(data[0]), batch):
        part = tuple(a[start:start + batch] for a in data)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, targets = query_logits(model, part)
            loss += F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), reduction='sum').item()
        pred = logits.argmax(-1)
        if return_predictions: predictions.append(pred.cpu().numpy())
        good = pred == targets
        correct += good.sum().item()
        exact += good.all(-1).sum().item()
        count += good.numel()
        examples += good.shape[0]
        auxiliary(model)
    model.train()
    metrics = dict(accuracy=correct/count, exact_accuracy=exact/examples, eval_loss=loss/count)
    return (metrics, np.concatenate(predictions)) if return_predictions else metrics


def query_blind_controls(task, data):
    x, positions, y = data
    _, value_base, _ = task_tokens(task)
    majority, last, oracle = [], [], []
    for tokens, pos in zip(x, positions):
        values = tokens[(tokens >= value_base) & (tokens < value_base + VALUES)]
        majority.append(np.bincount(values - value_base, minlength=VALUES).argmax() + value_base)
        last.append(values[-1])
        if task == 'state':
            latest = {int(tokens[j]): int(tokens[j+1]) for j in range(0, int(pos[0])-1, 2) if tokens[j] != PAD}
            final_values = np.asarray(list(latest.values()))
            oracle.append(np.bincount(final_values-value_base, minlength=VALUES).argmax()+value_base)
        else:
            oracle.append(majority[-1])
    return dict(query_blind_oracle_accuracy=float((np.asarray(oracle)[:, None] == y).mean()),
                majority_value_accuracy=float((np.asarray(majority)[:, None] == y).mean()),
                last_value_accuracy=float((np.asarray(last)[:, None] == y).mean()))


def binding_control(task, data, predictions):
    """Compare the same predictions with another queried entity/chain's value.

    For a query-independent strategy the expected gap is zero, even when
    context-value frequencies let it beat unconditional value chance.
    """
    x, _, targets = data
    correct = (predictions == targets).astype(float)
    if task == 'state':
        # Each row has four distinct queried entities in random order.
        matches = (predictions[:, :, None] == targets[:, None, :]).sum(-1)
        other = (matches - correct) / (targets.shape[1] - 1)
    else:
        value_base = task_tokens(task)[1]
        other = []
        for tokens, pred, good in zip(x, predictions[:, 0], correct[:, 0]):
            values = tokens[(tokens >= value_base) & (tokens < value_base + VALUES)]
            # One terminal assignment per disjoint chain; remove the queried one.
            other.append(((values == pred).sum() - good) / (len(values)-1))
        other = np.asarray(other)[:, None]
    return dict(other_entity_value_accuracy=float(other.mean()),
                binding_gap_pp=float(100*(correct-other).mean()))


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def run(args, task_api=None):
    # New generators can reuse the matched trainer without changing old tasks.
    task_tokens, sequence_length, generate, query_blind_controls, binding_control = (
        globals()[name] if task_api is None else getattr(task_api, name)
        for name in ('task_tokens', 'sequence_length', 'generate',
                     'query_blind_controls', 'binding_control'))
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    recipe = vars(args).copy()
    recipe.pop("max_minutes")
    recipe.update(layers=2, head_dim=16, state_dim=16, id_count=task_tokens(args.task)[0],
                  values=VALUES, chance=1/VALUES, sequence_length=sequence_length(args.task, args.load, args.updates, args.hops, args.min_length),
                  weight_decay=.1, optimizer='AdamW', scheduler='cosine to zero',
                  data_protocol='fresh independent per step; independent validation and final test',
                  metric='unrestricted full vocabulary argmax at query identifiers')
    if task_api is not None and hasattr(task_api, 'recipe_metadata'):
        recipe.update(task_api.recipe_metadata())
    recipe_path = folder/'recipe.json'
    if recipe_path.exists():
        previous_recipe = json.loads(recipe_path.read_text())
        previous_recipe.pop('max_minutes', None)
        previous_recipe.setdefault('min_length', 256)
        if previous_recipe != recipe:
            raise ValueError('Refusing to mix recipes in an existing run directory')
    atomic_json(recipe_path, recipe)
    old_result = json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
    if old_result and 'binding_gap_pp' in old_result:
        print('ALREADY_COMPLETE ' + str(folder), flush=True)
        return
    if old_result and not (folder/'checkpoint.pt').exists():
        raise ValueError('Evaluation backfill requires the completed checkpoint')
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = make_model(args.family, args.width, args.d, recipe['sequence_length'], task_tokens(args.task)[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    val = generate(args.task, args.load, args.eval_examples,
                   np.random.default_rng(np.random.SeedSequence([args.seed, 10001, args.load])), args.updates, args.hops, args.min_length)
    val_hash = hashlib.sha256(b''.join(a.tobytes() for a in val)).hexdigest()
    start_step = 0
    elapsed = 0.
    checkpoint = folder/'checkpoint.pt'
    if checkpoint.exists():
        ck = torch.load(checkpoint, weights_only=False, map_location='cpu')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start_step, elapsed = ck['step'], ck['elapsed']
        if old_result and start_step != args.steps:
            raise ValueError('Completed result has an incomplete checkpoint')
        for m in model.modules():
            if hasattr(m, '_steps'): m._steps = start_step
    started = time.monotonic()
    deadline = started + args.max_minutes * 60
    base = dict(task=args.task, load=args.load, family=args.family, d=args.d,
                width=args.width, seed=args.seed, parameters=params, trainable_parameters=trainable,
                validation_sha256=val_hash, sequence_length=recipe['sequence_length'])
    metrics_path = folder/'metrics.jsonl'
    if metrics_path.exists():
        # Drop log rows beyond the last durable checkpoint after an interrupted write.
        lines = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        metrics_path.write_text(''.join(json.dumps(row)+'\n' for row in lines if row['step'] <= start_step))
    print('START ' + json.dumps(base), flush=True)
    def save(step):
        tmp = folder/'checkpoint.tmp'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), step=step,
            elapsed=elapsed + time.monotonic()-started, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()), tmp)
        tmp.replace(checkpoint)
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        data = generate(args.task, args.load, args.batch,
            np.random.default_rng(np.random.SeedSequence([args.seed, 20001, args.load, step])), args.updates, args.hops, args.min_length)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, targets = query_logits(model, data)
            ce = (task_api.training_loss(logits, targets, data)
                  if task_api is not None and hasattr(task_api, 'training_loss')
                  else F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten()))
            loss = ce + auxiliary(model)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad):
            raise RuntimeError(f'Nonfinite gradient at step {step}')
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            record = dict(base, step=step, train_loss=ce.item(), grad_norm=grad.item(),
                lr=optimizer.param_groups[0]['lr'], elapsed_seconds=elapsed + time.monotonic()-started,
                **evaluate(model, val, args.batch))
            with metrics_path.open('a') as fp: fp.write(json.dumps(record)+'\n')
            atomic_json(folder/'status.json', record)
            print('METRICS '+json.dumps(record), flush=True)
            save(step)
        if time.monotonic() > deadline and step < args.steps:
            save(step)
            print('TIME_SLICE_COMPLETE', flush=True)
            return
    test = generate(args.task, args.load, args.test_examples,
        np.random.default_rng(np.random.SeedSequence([args.seed, 30001, args.load])), args.updates, args.hops, args.min_length)
    final_metrics, predictions = evaluate(model, test, args.batch, return_predictions=True)
    np.savez_compressed(folder/'test_predictions.npz', predictions=predictions, targets=test[2], positions=test[1])
    result = dict(base, steps=args.steps, complete=True, elapsed_seconds=elapsed+time.monotonic()-started,
        test_sha256=hashlib.sha256(b''.join(a.tobytes() for a in test)).hexdigest(),
        **final_metrics, **query_blind_controls(args.task, test),
        **binding_control(args.task, test, predictions))
    if task_api is not None and hasattr(task_api, 'extra_evaluation'):
        result.update(task_api.extra_evaluation(model, args, evaluate))
    if old_result:
        if result['test_sha256'] != old_result['test_sha256']:
            raise ValueError('Evaluation backfill changed the test data')
        # Preserve primary scores; record the new evaluation diagnostic separately.
        result = dict(old_result, **binding_control(args.task, test, predictions),
                      binding_eval_accuracy=final_metrics['accuracy'])
    atomic_json(folder/'result.json', result)
    print('RESULT '+json.dumps(result), flush=True)


def argument_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['state', 'variable'], default='state')
    p.add_argument('--load', type=int, required=True)
    p.add_argument('--family', choices=['mamba2', 'gdn', 'gdn_current'], required=True)
    p.add_argument('--d', type=int, default=1)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=.01)
    p.add_argument('--min-length', type=int, default=256)
    p.add_argument('--updates', type=int, default=3)
    p.add_argument('--hops', type=int, default=2)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--eval-examples', type=int, default=256)
    p.add_argument('--test-examples', type=int, default=1024)
    p.add_argument('--max-minutes', type=float, default=100)
    p.add_argument('--out', required=True)
    return p


if __name__ == '__main__':
    run(argument_parser().parse_args())
