"""Fail before training if Python resolves a frozen module over an experiment."""
import hashlib
import importlib
import json
import os
from pathlib import Path


def verify_sources(stage):
    expected = json.loads(os.environ['GDN_EXPECTED_SOURCES_JSON'])
    actual = {}
    for name, reference in expected.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual[name] = dict(path=str(path), sha256=digest)
        if str(path) != reference['path'] or digest != reference['sha256']:
            raise RuntimeError(f'Wrong runtime source for {name}: expected {reference}, got {actual[name]}')
    root = Path(os.environ['MQAR_RUN_DIR'])
    (root/f'{stage}-sources.json').write_text(json.dumps(actual,indent=2)+'\n')
    print('SOURCE_AUDIT '+json.dumps(dict(stage=stage,verified_modules=list(actual))),flush=True)
    return actual
