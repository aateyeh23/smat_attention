"""Freeze the six-run comparison before validation or training."""
import hashlib
import json
import shutil
from pathlib import Path
from common import BASE,CODE,DATA,ROOT,SEEDS,atomic_json

def main():
    assert not (ROOT/'source_sha256.json').exists(), 'Already frozen'
    tasks=[]
    for variant in ('mom_profiles','mom_bytes'):
        root=ROOT/variant;root.mkdir(exist_ok=True)
        if not (root/'data').exists(): (root/'data').symlink_to(DATA/'data',target_is_directory=True)
        shutil.copy2(DATA/'dataset_config.json',root/'dataset_config.json')
        for seed in SEEDS:
            folder=root/f'shared-gdn_current-d3-w64-s{seed}'
            tasks.append(dict(name=f'{variant}-s{seed}',variant=variant,seed=seed,
                root=str(root),result=str(folder/'result.json')))
    atomic_json(ROOT/'tasks.json',tasks)
    atomic_json(ROOT/'protocol.json',dict(
        parent_campaign=str(BASE),seeds=SEEDS,variants=['mom_profiles','mom_bytes'],
        mechanism='MoM Gated DeltaNet with independent per-memory K/V, beta and decay projections; shared Q and output; top4 reads/writes; no shared memory.',
        memory_matching='Per length match q^2 states or q^2+q(q+1) states to SMAT matrix tables. Count is per head/layer. Same 2 heads, key/value dim16, 2 layers.',
        auxiliary='Coefficient .01 times mean per-layer Switch/MoM balancing loss; includes all four route positions.',
        routing='One learned router per length, shared across heads as in MoM. Expert bank shared across lengths via prefixes. SMAT instead routes each head independently.',
        initialization='Construct paired SMAT then replace mixers; embedding, residual norms and tied output head retained. MoM mixer uses snapshotted FLA initialization.',
        training=dict(epochs=32,batch=256,lr=.003,weight_decay=.1,steps=22560,examples_per_epoch=180000,early_stopping=False),
        primary='Final test accuracy at 128 bindings, paired by training seed against geometry; all loads, macro and exact recall secondary.',
        limitations=['Whole-mixer baseline, not a geometry-only intervention. Four writes vs SMAT one.',
          'Matches provisioned FP32 matrix states only; MoM per-memory convolution cache, total GPU allocation, parameters and runtime separately reported.',
          'No shared memory and top4 are modifications to the paper default top2 plus shared.',
          'Many more memories than paper default; single fixed training recipe, no hyperparameter tuning.',
          'Trainer legacy family/d labels remain gdn_current/d3 for file compatibility; ablation.json identifies actual architecture.'],
        implementation='Snapshot FLA MomAttention initialization; own mathematically equivalent packed forward with stable chronological sort and per-stream convolution; independent batch shards with activation checkpointing.',
        upstream='https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/mom.py',
        paper='https://arxiv.org/abs/2502.13685',
        data_manifest_sha256=hashlib.sha256((DATA/'data/manifest.json').read_bytes()).hexdigest()))
    sources=json.loads((BASE/'source_sha256.json').read_text())
    for filename,digest in sources.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
    for path in [*CODE.glob('*.py'),*CODE.glob('*.sbatch'),ROOT/'tasks.json',ROOT/'protocol.json']:
        sources[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(ROOT/'source_sha256.json',sources)
    print('Frozen six MoM runs',flush=True)

if __name__=='__main__':main()
