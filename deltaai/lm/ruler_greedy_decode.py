"""Greedy decoding accelerated by verifying continuations from output n-grams.

Drafts use only already generated tokens. Each accepted token is checked against
the full model's greedy prediction; the first mismatch is replaced by that
prediction. No reference answers or early score-based stopping are used.
"""
import json
import time

import torch
from torch.nn import functional as F


def draft_continuation(history, limit):
    for n in range(min(8, len(history)-1), 1, -1):
        suffix = history[-n:]
        for begin in range(len(history)-n-1, -1, -1):
            if history[begin:begin+n] == suffix:
                return history[begin+n:begin+n+limit]
    return []


def verified_prefix(draft, predictions, remaining, eos):
    accepted = []
    for index, token in enumerate(predictions[:len(draft)+1]):
        accepted.append(token)
        if token == eos or len(accepted) == remaining:
            break
        if index == len(draft) or token != draft[index]:
            break
    return accepted


@torch.inference_mode()
def decode(model, batch, eos, budget=128, draft_limit=16):
    device = next(model.parameters()).device
    tokens = torch.full((len(batch), model.cfg.length), eos, dtype=torch.long, device=device)
    prompt_lengths = [len(row['input_ids']) for row in batch]
    for i, row in enumerate(batch):
        tokens[i, :prompt_lengths[i]] = torch.tensor(row['input_ids'], device=device)
    generated = [[] for _ in batch]
    ended = [False]*len(batch)
    forward_calls = 0
    started = time.monotonic()
    while not all(ended):
        drafts, lengths = [], []
        for i, history in enumerate(generated):
            remaining = budget-len(history)
            length = prompt_lengths[i]+len(history)
            lengths.append(length)
            draft = [] if ended[i] else draft_continuation(history, min(draft_limit, remaining-1))
            drafts.append(draft)
            tokens[i, length:] = eos
            if draft:
                tokens[i, length:length+len(draft)] = torch.tensor(draft, device=device)
        hidden, _ = model.hidden(tokens)
        span = max(map(len, drafts))+1
        positions = (torch.tensor(lengths, device=device)[:, None]-1+
                     torch.arange(span, device=device)[None, :]).clamp(max=model.cfg.length-1)
        logits = F.linear(hidden[torch.arange(len(batch), device=device)[:, None], positions],
                          model.embedding.weight).float()
        greedy = logits.argmax(-1).tolist()
        del hidden, logits
        if forward_calls == 0:
            reference = model.next_logits(tokens[:1, :lengths[0]]).argmax(-1).item()
            assert reference == greedy[0][0], 'Batched decoding differs from reference'
        forward_calls += 1
        for i in range(len(batch)):
            if ended[i]:
                continue
            accepted = verified_prefix(drafts[i], greedy[i], budget-len(generated[i]), eos)
            tokens[i, lengths[i]:lengths[i]+len(accepted)] = torch.tensor(accepted, device=device)
            generated[i].extend(accepted)
            ended[i] = accepted[-1] == eos or len(generated[i]) == budget
        if forward_calls % 16 == 0:
            print('VERIFIED_DECODE '+json.dumps(dict(forward_calls=forward_calls,
                generated_lengths=list(map(len, generated)), seconds=time.monotonic()-started)), flush=True)
    print('DECODE_TIMING '+json.dumps(dict(forward_calls=forward_calls,
        generated_tokens=sum(map(len, generated)), seconds=time.monotonic()-started)), flush=True)
    return generated
