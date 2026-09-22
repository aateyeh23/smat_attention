"""Isolated three-arm campaign for the existing adaptive MQAR dispatcher."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = Path(os.environ['MQAR_FOUR_READS_DIR'])
TASKS = [('fourreads', 16, d) for d in (2, 3, 4)]
RELEASE_JOB = None


def paths(task):
    group, width, d = task
    assert group == 'fourreads' and width == 16 and d in (2, 3, 4)
    return str(ROOT / 'zoo_gdn_four_reads_configs.py'), RUN, RUN / f'w{width}-d{d}.json'


def prepare_worker():
    from run_mqar_four_reads import validate
    validate()
