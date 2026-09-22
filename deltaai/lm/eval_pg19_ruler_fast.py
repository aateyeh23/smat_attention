"""Greedy official RULER evaluation of immutable, fixed-geometry PG19 models."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import time

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from ruler_greedy_decode import decode


def digest(path):
    with Path(path).open('rb') as fp:
        return hashlib.file_digest(fp, 'sha256').hexdigest()


def main(args):
    torch.set_num_threads(4)
    checkpoint = Path(args.checkpoint)
    assert checkpoint.name == 'model-tokens-0500006912.pt'
    manifest = json.loads((checkpoint.parent/'manifest.json').read_text())
    source_root = Path('/opt/pg19-rank64')
    for name, expected in manifest['source_sha256'].items():
        assert digest(source_root/name) == expected, f'Source mismatch: {name}'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['tokens'] == 500006912
    module_name = ('pg19_baselines' if manifest['d'] == 1 else
                   'gdn_smat_transport_scale' if manifest['family'] == 'gdn' else 'mamba_smat_scale')
    module = importlib.import_module(module_name)
    assert Path(module.__file__).resolve() == source_root/'lm'/(module_name+'.py')
    if manifest.get('kernel_optimization') == 'tiled_fp32_v1':
        import pg19_optimized_kernels
        pg19_optimized_kernels.enable()
    cfg = module.ScaleConfig(**saved['config'])
    assert cfg.length == 16384
    torch.manual_seed(cfg.seed)
    model = module.ScaleLM(cfg).cuda().eval()
    model.load_state_dict(saved['model'], strict=True)
    del saved
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    data_root = Path(args.data)
    metadata = json.loads((data_root/'dataset-manifest.json').read_text())
    assert digest(data_root/'encoded.jsonl') == metadata['encoded_sha256']
    rows = [json.loads(line) for line in (data_root/'encoded.jsonl').read_text().splitlines()]
    for row in rows:
        assert row['input_ids'] == tokenizer.encode(row['input']+row['answer_prefix'], add_special_tokens=False)
        assert len(row['input_ids']) + metadata['max_new_tokens'] <= cfg.length
    if args.limit:
        rows = rows[:args.limit]
    metric_spec = importlib.util.spec_from_file_location('ruler_metric', data_root/'ruler-metrics.py')
    metric = importlib.util.module_from_spec(metric_spec)
    metric_spec.loader.exec_module(metric)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(arm=checkpoint.parent.name, checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint), training_tokens=500006912,
        model_source_sha256=manifest['source_sha256'], dataset=metadata,
        evaluator_sha256=digest(__file__), decoder_sha256=digest(Path(__file__).with_name('ruler_greedy_decode.py')), decoding='greedy', eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=128, fixed_model_length=cfg.length, samples=len(rows))
    if args.reference:
        reference_root = Path(args.reference)
        references = [json.loads(line) for line in (reference_root/'predictions.jsonl').read_text().splitlines()]
        assert len(references) >= 4, 'Need four reference outputs for the GPU equivalence gate'
        reference_manifest = json.loads((reference_root/'manifest.json').read_text())
        for key in ('checkpoint_sha256','dataset','training_tokens','model_source_sha256'):
            assert reference_manifest[key] == provenance[key], f'Changed reference {key}'
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            checked = decode(model, rows[:4], tokenizer.eos_token_id)
        assert checked == [row['generated_ids'] for row in references[:4]], 'Speculative/reference token mismatch'
        print('EQUIVALENCE_PASSED '+json.dumps(dict(examples=4, tokens=sum(map(len,checked)))), flush=True)
        provenance['reference_evaluator_sha256'] = reference_manifest['evaluator_sha256']
        if args.verify_only:
            (out/'verification.json').write_text(json.dumps(dict(passed=True, examples=4,
                generated_tokens=sum(map(len,checked)), provenance=provenance), indent=2)+'\n')
            return
    provenance_path = out/'manifest.json'
    if provenance_path.exists():
        assert json.loads(provenance_path.read_text()) == provenance, 'Changed evaluation recipe'
    provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
    predictions_path = out/'predictions.jsonl'
    if args.reference and not predictions_path.exists():
        predictions_path.write_text(''.join(json.dumps(row)+'\n' for row in references))
        (out/'migration.json').write_text(json.dumps(dict(reference=str(reference_root),
            reference_predictions_sha256=digest(reference_root/'predictions.jsonl'),
            reused_examples=len(references), equivalence_examples=4), indent=2)+'\n')
    predictions = ([json.loads(line) for line in predictions_path.read_text().splitlines()]
                   if predictions_path.exists() else [])
    assert [r['sample_id'] for r in predictions] == list(range(len(predictions)))
    if len(predictions) == len(rows):
        print('RESULT '+(out/'result.json').read_text(), flush=True)
        return
    started = time.monotonic()
    initial_count = len(predictions)
    batch_size = args.batch
    first_batch = True
    while len(predictions) < len(rows):
        batch = rows[len(predictions):len(predictions)+batch_size]
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                generated = decode(model, batch, tokenizer.eos_token_id)
        except torch.cuda.OutOfMemoryError:
            # A failed batch has written no predictions; retry it with less memory.
            if batch_size == 1:
                raise
            (out/'retry-batch.json').write_text(json.dumps(dict(batch=max(1, batch_size//2))))
            raise
        first_batch = False
        with predictions_path.open('a') as fp:
            for row, ids in zip(batch, generated):
                pred = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                processed = re.sub(r'[\x00-\x1f]', '\n', pred.strip()).strip()
                score = metric.string_match_all([processed], [row['outputs']])
                result = dict(sample_id=row['sample_id'], index=row['index'], outputs=row['outputs'],
                    pred=pred, generated_ids=ids, input_length=len(row['input_ids']),
                    token_position_answer=row['token_position_answer'], score=score)
                fp.write(json.dumps(result)+'\n')
                predictions.append(result)
        score = metric.string_match_all(
            [re.sub(r'[\x00-\x1f]', '\n', r['pred'].strip()).strip() for r in predictions],
            [r['outputs'] for r in predictions])
        progress = dict(arm=checkpoint.parent.name, completed=len(predictions), total=len(rows),
                        score=score, elapsed_seconds=time.monotonic()-started,
                        seconds_per_example=(time.monotonic()-started)/(len(predictions)-initial_count),
                        batch_size=batch_size, gpu_peak_gb=torch.cuda.max_memory_allocated()/1e9,
                        complete=len(predictions) == len(rows))
        for half in ('first_half', 'second_half'):
            subset = [r for r in predictions if (r['token_position_answer'] < 8192) == (half == 'first_half')]
            progress[half] = dict(n=len(subset), score=sum(r['score'] for r in subset)/len(subset) if subset else None)
        (out/'status.json').write_text(json.dumps(progress, indent=2)+'\n')
        print('PROGRESS '+json.dumps(progress), flush=True)
    (out/'result.json').write_text(json.dumps(progress, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    main(parser.parse_args())
