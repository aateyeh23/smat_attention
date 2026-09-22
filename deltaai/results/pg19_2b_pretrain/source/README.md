This snapshot accompanies the 2B-token PG19 checkpoints. `source-bundle.tar.gz`
contains the frozen base model/FLA/Zoology sources plus these optimized overrides.
Runtime versions and build commands are in `modal_pg19_pretrain.py` and the
repository's `modal_mqar_four_reads.py`: CUDA 13, PyTorch 2.11, Triton 3.7.1.

After extracting the bundle, add these directories to PYTHONPATH:
`optimized:optimized/lm:base/deltaai:base/smat:base/zoology:base/fla`.
Use the same CUDA runtime/dependencies as the recorded Modal image.

```python
import torch
from gdn_smat_scale import ScaleConfig, ScaleLM

saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model = ScaleLM(ScaleConfig(**saved["config"])).cuda().eval()
model.load_state_dict(saved["model"])
# input_ids: GPT-2 token IDs, CUDA int64, shape [batch, prompt_length].
with torch.autocast("cuda", dtype=torch.bfloat16):
    output_ids = model.generate(input_ids, max_new_tokens=64, eos_token_id=50256)
```

Generation retains the fixed 16K training geometry and recomputes the window
for each token. Prompt plus answer must fit inside 16,384 tokens. It has no
generation cache and does not extrapolate beyond the trained window.

`latest.pt` includes model, optimizer, RNG, progress and recipe for resuming.
`model-tokens-*.pt` retains model/config/recipe/validation at each 250M-token
milestone and at completion. Metrics are in each arm's `metrics.jsonl`;
the same 64 validation windows are evaluated every 50M tokens, and the full
280-window validation set is evaluated at completion. Test data is held out.
