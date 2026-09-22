# Joint-recall development trials

User objective: iterate until both current SMat variants beat the native backbones;
one seed (123) is sufficient. All trials, including unsuccessful ones, are retained.
Task changes use matched inputs, training exposure and validation splits for every model.
Validation drives iteration. Existing test results are development evidence; the final
selected setting requires a new independently generated confirmation test.

1. `../joint_recall`: original block context, five table sizes,180k examples,32 epochs.
   Resume seed123 only; the user cancelled the planned three-seed expansion.
2. `explicit_context_c8_k16`: explicit context on every key/value record and every query;
   shuffled record/query order,8 contexts ×16 keys =128 bindings,180k examples,32 epochs.
   This removes context persistence across blocks and increases associative-memory load.
   Shared keys require the conjunction of context and key. Native and SMat architectures
   are unchanged. Both models see identical tokens and labels. PAD masks all query answers.
   Data seed20260919; validation5000, test10000. This is a new composite-key recall variant,
   not the published block-context encoding. It tests one load to avoid length-specific
   router parameters receiving very different amounts of training in a mixed-size dataset.

3. `explicit_context_mixed`: same explicit-record encoding, with60k examples
   at each of C1/K4, C2/K8, C8/K16 (180k total). The smallest cell is ordinary
   key retrieval with one context; the others require context+key binding.
   All models receive the same interleaved batches for32 full epochs. Primary
   development comparisons will report every cell, including C8/K16; no cells
   are discarded based on results. Data seed20260920, validation2000/cell,
   test4000/cell. Motivation: the large-only native GDN remained at chance
   through14 epochs, and first-layer scalar retention probes showed severe
   forgetting in both GDN variants. This changes task sampling, not architecture.

4. `explicit_context_c8_k16_longinit`: exactly the same cached data and32-epoch
   schedule as trial2. Both GDN variants initialize their existing scalar forget
   gates with tau=4*772=3088: dt_bias=log(expm1(1/(tau*exp(A_log)))). This makes
   alpha=exp(-1/tau) when the learned input projection is zero. All weights remain
   trainable and the architectures, parameter counts, LR0.01 and training data
   are unchanged. Apply identically to native and SMat GDN. Motivation: the
   default native GDN completed32 epochs at6.29% validation, and direct scalar
   retention probes showed severe forgetting. This is a training-initialization
   comparison, not a change to the SMat mask or recurrence equations.

5. `explicit_context_mixed_lr003`: same cached data as trial3, architectures and
  32-epoch budget; LR0.003 for all four models. Other hyperparameters are unchanged.
   Motivation: native GDN learns on the mixed task, while SMat GDN remains near
   chance at epoch11. Large-only SMat Mamba also regressed after early gains,
   suggesting that optimizer stability deserves a controlled check. A claimed
   SMat advantage must also be compared against the strongest completed native
   validation result among the tested training settings on identical data.

6. `explicit_context_c8_k16_stableinit`: identical cached data to trial2, LR0.01
   and32 epochs. Weight decay is0 for all four architectures; both GDN variants
   additionally use the same tau3088 initialization from trial4. Mamba uses its
   unchanged initialization. Compared with trial4, the GDN contrast isolates
   weight decay. Compared with trial2, the Mamba contrast isolates weight decay.
   The failed mixed-task SMat GDN had near-zero first-layer A_log/dt_bias/a_proj
   weights by epoch20 and scalar retention approximately0.5 per token, whereas
   the matched native model learned near-unit retention. This motivates the
   optimizer ablation; it does not by itself prove why learning initially failed.
   Best native validation across tested settings on the same data remains the
   comparison standard when assessing a positive SMat result.

7. `explicit_context_mixed_stableinit`: both GDN variants use the exact cached
   mixed dataset (trial3), with tau3088 and weight decay0 as in trial6. Other
   parameters and32-epoch exposure remain fixed. This combines the two controlled
   interventions: learn first from short tables while keeping memory available.
   The large-only improved initialization helped native GDN, but SMat remained
   at chance at epoch11; scalar retention alone has not solved learning so far.

8. `explicit_context_c8_k16_no_scalar_decay` and
   `explicit_context_mixed_no_scalar_decay`: separately labeled SMat GDN ablations
   of trials2 and3. Set the existing `transport_scalar_decay` option to false;
   preserve delta-transition transport, native/local GDN dynamics, routing,
   initialization, LR0.01, weight decay0.1, and32 epochs. Reuse the exact cached
   data and already measured native controls. This is a memory-operator ablation,
   not an unchanged-architecture claim. Motivation: direct forward probes found
   zero median transported keys/queries in the failed SMat GDN checkpoint, and
   inspection confirmed that SMat Mamba already disables time decay in its extra
   memory. The result isolates scalar decay; delta transitions still differ from
   the Mamba memory. Compare against the strongest completed native setting on
   identical data, including the improved GDN optimizer/initialization controls.

9. `explicit_context_capacity512`: a capacity sweep following the user's question
   whether the 128-binding task is too easy. Keep the explicit-record encoding and
   test 4,16,128,256,512 bindings: C1/K4, C2/K8, C8/K16, C16/K16, C32/K16.
   Use36,000 fixed training tables per cell (180,000 total),2,000 validation and
   4,000 development-test tables per cell; data seed20260922. Every model receives
   all five sizes, with the same batch order and32-epoch budget. The larger key
   vocabulary has512 IDs so the paired unique-key control can represent512 entries;
   all models use vocabulary563 at every size. Thus the128-binding points here
   must be rerun, not spliced together with previous datasets' scores.
   Models retain their original architecture and memory settings, including
   scalar decay in SMat GDN. Start with LR0.003 for all four models; also test
   native LR0.01 controls before claiming an advantage. Weight decay remains0.1.
   Before full training, run a200-update pilot at width64 and batch32 on all four
   models in `capacity512_pilot`; pilot checkpoints are separate from the main
   batch256 runs. The entire load curve and earlier negative comparisons remain
   reportable. This tests capacity limits; it does not establish that task size
   caused the previous Mamba gap, nor does it erase that negative result.
   Once both earlier mixed-size Mamba runs finished, add SMat Mamba LR0.01 on
   the same capacity data: that was its stronger prior recipe. Both native
   learning rates remain controls; all capacity settings receive32 full epochs.

10. All-d extension of `explicit_context_capacity512`: after the completed d=3
    validation comparison passed the conditional gate, add d=2 and d=4 for both
    backbones at the user's request. Use the same frozen data, source, LR0.003,
    weight decay0.1, width64, two layers, seed123, and32-epoch budget. Report all
    eight primary models regardless of outcome. The separate native GDN LR0.01
    checkpoint joins fresh evaluation because it is stronger at high load than
    the LR0.003 native setting. `confirmation_plan.json` fixes these nine folders
    before fresh evaluation; `CAPACITY_PROTOCOL.md` retains the prespecified d=3
    criterion rather than selecting a new dimension from the fresh results.

11. `explicit_context_capacity512_lr004`: user-requested learning-rate follow-up
    for SMat GDN d=2. Run native GDN and SMat GDN d=2 at LR0.004, retaining seed123,
    the exact cached capacity data, frozen source_v5, and the full32-epoch budget.
    The LR0.003 d=2 model learns short tables and improves late on long tables;
    a modest LR increase tests slow learning under the fixed budget. No causal
    explanation is assumed. See the local `PROTOCOL.md` for the fixed comparison.
    These are additional development runs; the original common-LR all-d results
    and nine-model fresh confirmation remain intact.
