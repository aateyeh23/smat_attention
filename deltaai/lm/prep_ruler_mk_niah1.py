"""Generate official RULER MK-NIAH-1 with GPT-2 tokenization and provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    import html2text
    from bs4 import BeautifulSoup
    import tiktoken
    import yaml

    repo, out = args.repo.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = repo / 'scripts/data/synthetic'
    essay_path = source / 'json/PaulGrahamEssays.json'
    urls = (source / 'json/PaulGrahamEssays_URLs.txt').read_text().splitlines()
    def fetch(url):
        # HTTPS and the raw GitHub host serve the same resources without redirects.
        resolved = url.replace('http://', 'https://').replace(
            'https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/',
            'https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/main/')
        with urllib.request.urlopen(resolved, timeout=60) as response:
            raw = response.read()
        is_html = '.html' in url
        if is_html:
            converter = html2text.HTML2Text()
            converter.ignore_images = converter.ignore_tables = converter.escape_all = True
            converter.reference_links = converter.mark_code = False
            text = converter.handle(str(BeautifulSoup(raw.decode('unicode_escape'), 'html.parser').find('font')))
        else:
            text = raw.decode('utf-8')
        return (is_html, url.rsplit('/', 1)[-1], text,
                dict(url=url, resolved_url=resolved, sha256=hashlib.sha256(raw).hexdigest()))
    if not essay_path.exists():
        with ThreadPoolExecutor(max_workers=12) as pool:
            essays = list(pool.map(fetch, urls))
        essays.sort(key=lambda row: (row[0], row[1]))
        essay_path.write_text(json.dumps(dict(text=''.join(row[2] for row in essays))))
        (out / 'essay-downloads.json').write_text(json.dumps([r[3] for r in essays], indent=2))
    spec = importlib.util.spec_from_file_location('ruler_data_constants', source/'constants.py')
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    task = yaml.safe_load((repo/'scripts/synthetic.yaml').read_text())['niah_multikey_1']
    base = constants.TASKS['niah']
    command = [sys.executable, str(source/'niah.py'), '--save_dir', str(out),
               '--save_name', 'niah_multikey_1', '--tokenizer_type', 'openai',
               '--tokenizer_path', 'gpt2', '--max_seq_length', '16384',
               '--tokens_to_generate', str(base['tokens_to_generate']),
               '--num_samples', str(args.samples), '--random_seed', '42',
               '--template', base['template'] + base['answer_prefix']]
    for key, value in task['args'].items():
        command.extend(['--'+key, str(value)])
    subprocess.run(command, check=True)
    dataset = out/'niah_multikey_1/validation.jsonl'
    tokenizer = tiktoken.get_encoding('gpt2')
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    assert len(rows) == args.samples
    lengths = []
    for i, row in enumerate(rows):
        ids = tokenizer.encode(row['input']+row['answer_prefix'])
        assert len(ids)+128 <= 16384
        assert len(row['outputs']) == 1 and row['outputs'][0] in row['input']
        row.update(sample_id=i, input_ids=ids)
        lengths.append(len(ids))
    encoded = out/'encoded.jsonl'
    encoded.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    manifest = dict(task='niah_multikey_1', ruler_commit=subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        samples=len(rows), seed=42, tokenizer='gpt2', max_seq_length=16384,
        max_new_tokens=128, template='base', task_args=task['args'],
        input_length_min=min(lengths), input_length_max=max(lengths),
        dataset_sha256=digest(dataset), encoded_sha256=digest(encoded),
        essay_sha256=digest(essay_path), source_sha256={str(p.relative_to(repo)):digest(p)
            for p in [source/'niah.py', source/'constants.py', repo/'scripts/synthetic.yaml',
                      repo/'scripts/eval/synthetic/constants.py']})
    (out/'dataset-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=500)
    main(parser.parse_args())
