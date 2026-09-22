"""Prepare a fresh PG19 corpus on CPU and persist it for all four GPU arms."""
from pathlib import Path
import modal

LOCAL = Path(__file__).resolve().parent
app = modal.App('smat-pg19-scale-data')
volume = modal.Volume.from_name('smat-pg19-scale-data-v1', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy==1.26.4', 'tiktoken==0.12.0', 'requests==2.32.5')
         .add_local_file(LOCAL/'lm/prep_pg19_scale.py', '/root/prep_pg19_scale.py'))


@app.function(image=image, cpu=16, memory=32768, timeout=14400,
              volumes={'/data': volume})
def prepare():
    from prep_pg19_scale import prepare as prepare_corpus
    result = prepare_corpus('/data/pg19-16k-1b')
    volume.commit()
    return result


@app.function(image=image, cpu=16, memory=32768, timeout=14400,
              volumes={'/data': volume})
def extend():
    from prep_pg19_scale import prepare as prepare_corpus
    volume.reload()
    result = prepare_corpus('/data/pg19-16k-2b', target_tokens=2_000_000_000,
                            extend_from='/data/pg19-16k-1b')
    volume.commit()
    return result
