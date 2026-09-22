# Portable PG19 checkpoints

These are partially trained checkpoints, stopped at the user’s request before
reaching the 2B-token target. `checkpoints.json` records each exact saved token
count. The four models have different training progress; compare validation
scores at matched tokens using the included histories.

| Model | Saved training tokens | Weights file |
|---|---:|---|
| GDN | 877,002,752 | [d1/weights.pt](d1/weights.pt) |
| GDN + SMAT d=2 | 540,672,000 | [d2/weights.pt](d2/weights.pt) |
| GDN + SMAT d=3 | 500,039,680 | [d3/weights.pt](d3/weights.pt) |
| GDN + SMAT d=4 | 405,667,840 | [d4/weights.pt](d4/weights.pt) |

All checkpoint downloads passed SHA-256 verification. The Modal app is stopped
with zero tasks; original checkpoint and dataset volumes are retained.

## Files

- `d1`: GDN; `d2`, `d3`, `d4`: GDN + SMAT at the respective d.
- `d*/weights.pt`: model tensors, architecture config, training recipe and progress.
- `d*/resume.pt`: full checkpoint, including optimizer, RNG, data cursor and scheduler progress.
- `d*/manifest.json`: SHA-256 and byte counts verified against the Modal export.
- `source-bundle.tar.gz`: frozen model sources and optimized kernels.
- `runtime-pip-freeze.txt`: packages recorded from the actual training container.
- `data-manifest.json`: training data provenance and checksums. Dataset binaries
  are not included here; they remain in Modal volume `smat-pg19-scale-data-v1`.

## Run on another machine

Copy this entire directory. Use a compatible NVIDIA CUDA machine; the optimized
path was validated on H100, not on arbitrary GPU architectures. Training used
Python 3.12, CUDA 13.0, PyTorch 2.11.0 and Triton 3.7.1. Extract the source:

```bash
tar -xzf source-bundle.tar.gz
export PYTHONPATH="$PWD/optimized:$PWD/optimized/lm:$PWD/base/deltaai:$PWD/base/smat:$PWD/base/zoology:$PWD/base/fla"
export FLA_TILELANG=0
export TRITON_F32_DEFAULT=tf32x3
```

`base-image-builder.py` records exact dependency installation/build commands.
Recreate those package versions with CUDA-compatible builds on the destination;
its Modal-specific source mounts are already supplied by the extracted bundle.
The pip freeze is an environment record, not a relocatable requirements file
(local kernel build paths in it refer to the original container).

```python
import torch
from gdn_smat_scale import ScaleConfig, ScaleLM

saved = torch.load("d3/weights.pt", map_location="cpu", weights_only=False)
model = ScaleLM(ScaleConfig(**saved["config"])).cuda().eval()
model.load_state_dict(saved["model"], strict=True)
# input_ids must contain GPT-2 token IDs, CUDA int64, shape [batch, prompt_length].
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    output_ids = model.generate(input_ids, max_new_tokens=64, eos_token_id=50256)
```

Generation uses the fixed 16,384-token training geometry and recomputes the
window each token. Prompt plus answer must fit within that length. There is no
generation cache. Model weights alone do not provide a Hugging Face AutoModel;
the bundled custom model code is required.

To generate from a text prompt after preparing the environment:

```bash
python generate.py --checkpoint d3/weights.pt --prompt "The old house stood" --max-new-tokens 32
```

This loads the GPT-2 tokenizer from Hugging Face on first use.

## Continue training

Download `/pg19-16k-2b` from Modal volume `smat-pg19-scale-data-v1`, retaining its
manifest and binary files. Verify their hashes against `data-manifest.json`.
For a given arm, copy `d3/resume.pt` to a new output directory as `latest.pt`
and copy `d3/recipe.json` alongside it. Then use the original recipe:

```bash
python optimized/lm/train_pg19_scale.py --d 3 --data /path/to/pg19-16k-2b --output /path/to/resume-d3 --batch 3 --global-batch-rows 8
```

Change `--d` and the corresponding checkpoint together. The original token
schedule and accumulated elapsed time are restored; `--max-hours` defaults to
11.8 cumulative hours, so increase it explicitly if extending that time budget.
Keep model, data and optimizer recipe settings unchanged for a matched resume.

Modal retains independent copies at volume `smat-pg19-2b-checkpoints`,
`/seed123/d{1,2,3,4}/handoff/`. Stopping the app does not delete the volumes.
