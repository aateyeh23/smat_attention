# Width-32 GDN + SMAT d=3 iterative search

**Current selected configuration:** `23_decay_no_rescale`, adopted by user
request. Scalar transport decay is enabled and incidence rescaling is disabled;
the other verified routing fixes remain enabled. Trial23 resumed after epoch14
and completed32epochs at82.78% versus GDN69.86%, passing all15 gates. See
`23_decay_no_rescale/FINAL.md`. Trial22 remains the archived prior configuration
at81.78%; selecting trial23 does not replace its results. Width32 d2/d4 and
width64 d2/3/4 are tracked in `decay-no-rescale-sweep.json` and
`decay-no-rescale-w64-sweep.json`. Width32/d2 and the entire width64 batch have
baseline stopping disabled by explicit user request.

**Source-resolution audit:** trials01–21 did not consistently execute their advertised router changes. See `SOURCE-RESOLUTION-AUDIT.md`. Their raw scores remain archived, but the prior architectural attributions are unverified. Verified-source training begins at trial22.

Goal: beat the existing width32 GDN baseline with task-general changes, keeping
two heads, head/state16, two layers, one hard bucket write and four distinct
weighted hyperplane reads. No full unreset GDN recurrence is added across the
SMAT boundary. No labels or MQAR-specific key/value offsets are added. The
inherited geometry has two fixed-position anchor writes; candidates03 and later
disable them (see `02_isolated_hash/address-alignment.md`). Candidate data,
seed123 and 32-epoch budget match the baseline.

## Mandatory stopping rule

At epochs 4,6,8,...,32, candidate overall validation accuracy must strictly
exceed GDN at the same epoch. A tie or lower score saves a resumable checkpoint
and terminates that trial before another training epoch. Epochs 1–3 and odd
epochs are observations, not stop gates. A trial's `complete=true` indicates
that its execution terminated; consult `result.json`/`gates.jsonl` to distinguish
rejection from a successful epoch32 completion.

The baseline is frozen from `mqar_width_sweeps/gdn/w32-d1-history.jsonl`.
Its epoch4 accuracy is 60.65875%, and its epoch32 accuracy is 69.8621875%.
`zoo_gdn_gated_trainer.py` implements the gate inside the training process.
`collect_gdn_iteration.py` collects comparisons and terminal checkpoints.

## Trials

1. `01_no_scalar_decay`: retain the restored write-hash gradients and rank-one
   transport, but remove scalar forgetting from the cross-boundary transport.
   Local GDN forgetting remains learned. Motivated by the previous run's severe
   loss of distant memory in one first-layer head. This is a general separation
   of local forgetting from distant profile retention, not an MQAR-specific rule.
   **Rejected at epoch4: 42.2528% vs GDN 60.6588%.** Checkpoint downloaded;
   verified that epoch5 did not run.
2. `02_isolated_hash`: retain no-scalar-decay transport, and detach the shared
   input to the learned write-hash convolution plus the beta weights used only
   in the occupancy auxiliary loss. The hash and convolution remain trainable;
   actual memory writes still train beta. Forward outputs are unchanged by this
   isolation. GPU checks passed; current app/path/status are in `current.json`.

Trial 2 was rejected at epoch 4: 56.66375% vs GDN 60.65875%. Its resumable
checkpoint, training history, and logs are archived locally. The user paused
for analysis, then explicitly resumed the search and requested automatic
iteration without asking to resume between candidates.

3. `03_content_only`: disable the two inherited fixed-position writes; keep
   every other trial02 setting. **Rejected at epoch4: 56.6434% vs 60.6588%.**
   The 16-pair score improved to61.00% from55.54%, but aggregate accuracy was
   unchanged because longer cases regressed. Anchor removal alone is insufficient.
4. `04_shared_address`: keep trial03's all-content writer. Score read planes
   by their probability mass under the writer's continuous query address,
   with Gaussian smoothing of .75 bins and the same learned W/gamma/b used
   by keys and queries. Share those parameters between lengths. Select and
   read exactly four distinct plane matrices. Dense scalar address scores do
   not imply additional memory reads. There is no cross-boundary value-state
   recurrence. The old independent categorical read parameters are frozen.
   Current app/path/status and next action are in `current.json`.
   **Rejected at epoch4:43.7866%.** First-layer hash directions became almost
   identical (cosines0.9998 and0.9984).
5. `05_shared_hash`: retain only cross-length hash sharing from trial04 and
   restore the categorical reader. **Rejected at epoch4:43.3763%.** Sharing
   alone also regresses, so it is not retained in the next candidate.
6. `06_tied_content`: start from trial03, with length-specific hashes and the
   categorical four-read selector. Transport uses tied memory query/key
   projections, with keys from the same learned causal context as writer
   addresses. Retrieval gradients now train that context convolution directly.
   Local GDN remains unchanged; the memory still uses identity-initialized
   operator scans, one content write, four reads, and no scalar erasure.
   The short memory-key convolution resets at the existing segment boundary.
   GPU checks passed, including gradients to the newly active memory projection.
   **Rejected at epoch4:41.8584%.**
7. `07_incidence_scale`: return to trial03 and multiply the memory branch by
   the number of cosets before its learned gate. This compensates the1/q
   expected inclusion weight under uniform plane selection; learned routing
   itself is not claimed to be an unbiased estimator. **Passed epoch4:
   61.0272% vs60.6588%; rejected epoch6:60.3394% vs64.4825%.**
8. `08_orthogonal_hash` was technically aborted: PyTorch's Householder
   parametrization stores sign metadata in parameter entries that AdamW decays.
   Its accuracy is not an architectural comparison. `08b_orthogonal_hash`
   replaces that parametrization with Cayley coordinates, preserves the
   initialization RNG stream, and verifies orthogonality after an AdamW update.
   This candidate otherwise retains trial07's incidence scaling and routing.
   **08b rejected at epoch4:52.52%.**
9. `09_gdn_key_hash`: retain trial07, but hash local GDN keys back-projected
   with the existing key projection. **Rejected at epoch4:54.8463%.** Reusing
   retrieval features for routing did not improve the comparison.
10. `10_address_reader`: isolate address-derived scoring from cross-length
    sharing, retaining incidence scaling. **Rejected at epoch4:37.0013%.**
    Immutable per-epoch checkpoint archives begin with this launcher version.
11. `11_symmetric_write_grad`: retain trial07's hard forward and categorical
    reader, but replace one-sided adjacent-bin write gradients with a local
    Gaussian surrogate centered on the hard bucket and both neighboring bins.
    At d3 this considers nine bucket gradients in backward; forward still pools
    one hard write and gathers four read matrices. Payload gradients and the
    existing occupancy loss are unchanged. Dense-reference and fixed-forward
    GPU checks are required before training.

Passing this search establishes the requested MQAR comparison on the specified
seed/protocol; it does not establish gains on selective copying or NLP without
separate downstream experiments. Repeated use of this validation set for model
selection must be disclosed in any reported generalization claim.

Trial11 was rejected at epoch4:53.71094% versus60.65875%. Its forward-preserving symmetric surrogate did not improve the comparison; terminal checkpoint and logs are archived locally.

12. `12_read_exploration`: From trial07: add Gaussian read-logit exploration with standard deviation linearly decaying from 1 to 0 over the existing 1000-step annealing period. Exactly four distinct weighted reads at every step; evaluation always deterministic. Same one hard content write, transport and incidence scaling. Motivation: checkpoint07 oracle-read intervention improves 32-pair sample accuracy from30.49% to54.98%, while ordinary plane target mass is near chance. No labels enter training or routing.

Trial12 rejected at epoch4:53.905%.

13. `13_joint_hash_balance`: From trial07: add total correlation (KL of joint soft write occupancy against the product of its coordinate marginals) to existing marginal hash balance, using the existing .01 balance coefficient. Joint counts aggregate the training batch per head; beta only weights scalar counts and remains detached for the auxiliary loss. No additional payload writes or reads. This penalizes diagonal collapse missed by separate coordinate histograms while leaving the hard forward and task-gradient surrogate unchanged.

Trial13 rejected at epoch4:52.35844%.

14. `14_tied_additive`: Return to trial07 scaling and categorical reading; use tied memory Q/K with learned causal key context (as trial06), but remove cross-boundary transition products. Profiles store additive beta*k*v with one hard content bucket per token; queries use the tied current-token feature and four weighted hyperplane reads. Local GDN resets and trains normally. The causal convolution is trained by retrieval itself, not only the write-gradient surrogate. No full value-state recurrence crosses the boundary. This tests the simpler additive alternative to transported tied features; it changes two settings relative to trial07.

Trial14 rejected at epoch4:35.44875%.

15. `15_aligned_additive`: From trial14: derive query hyperplane scores from the shared writer address map rather than an independent categorical read projection. This combines tied causal retrieval features, which train the writer context through the task loss, with address-aligned reading and simple additive profiles. Earlier shared-address trials did not use tied causal memory features without transport. Geometry and hash remain length-specific. Exactly one hard write and four distinct weighted reads.

Trial15 rejected at epoch4:39.46625%.

16. `16_periodic_hash`: From trial07: replace clamping of the upper interpolation neighbor with modulo-q wraparound, matching the cyclic finite-field coordinates. For uniform continuous coordinates, the old soft marginal occupancy has expected mass1/(2q) in bin0 and3/(2q) in bin(q-1), despite uniformly distributed hard writes. Wraparound gives1/q in every bin and avoids canceling the task routing gradient in the final bin when both neighbors coincide. Hard forward routing is unchanged; only the surrogate task gradient and auxiliary occupancy change. Exactly one write and four weighted reads.

Trial16 rejected at epoch4:54.81781%.

17. `17_profile_delta`: From trial14: replace additive profile accumulation with bucket-local delta correction S_c <- S_c + beta*k*(v-k^T S_c)^T, retaining normalized tied causal features, categorical four-plane reading, and incidence scaling. One hard forward write. The custom backward replays neighboring profiles only to recover the full hard-STE route gradient; payload/beta gradients come once through the hard forward. The actual write beta remains trainable while its occupancy-loss weight is detached. No full value-state recurrence crosses the boundary. Requires FP64 sequential-state reference and all-gradient GPU validation.

Trial17 rejected at epoch4:57.29%.

18. `18_aligned_profile_delta`: From trial17: replace categorical read scores with query hyperplane mass under the shared learned writer address map. Keep bucket-local delta updates and their full hard-STE backward, tied causal memory features, separate hashes per length, and incidence scaling. One physical write and four distinct weighted reads. Motivated by the21.84-point epoch4 improvement of profile delta over additive memory, and by the previously untested combination of corrected profile dynamics with aligned reading.

Trial18 rejected at epoch4:47.77438%.

19. `19_clipped_transport`: Use exactly trial07 architecture and add global gradient-norm clipping at1 before each AdamW update. This is an explicit training-recipe change; data, initialization seed, optimizer type, learning rate, decay, scheduler, epoch budget and strict baseline gates remain fixed. Hard routing and its surrogate can produce variable gradients; clipping limits rare updates that could dominate Adam moments. Record unclipped gradient norm mean/p95/max and clipping fraction every epoch. There is no claim yet that gradients are the cause or that clipping improves accuracy.

Trial19 rejected at epoch4:56.51063%.

20. `20_concurrent_profile_delta`: From trial18: restrict the four weighted read planes to those containing the query hard-hash cell. The top4 now have distinct directions and a common point, guaranteeing unit total inclusion weight for a matching writer cell. Scores remain learned through the shared smooth address distribution; plane selection and the hard query cell remain discrete. One payload write and four payload reads; no extra counterfactual read matrices. Keep profile delta, tied causal features, full write-gradient replay, and incidence scale; restore the original unclipped optimizer recipe. Motivated by trial17 second-layer target hash matches90.89% but per-head coverage52.86% at16pairs, with substantial collapse explicitly acknowledged.

Trial20 rejected at epoch4:49.47813%.

21. `21_slow_routing`: Use trial07 architecture and an explicit AdamW routing group at .1 times the main learning rate: hash parameters, categorical reader parameters, and causal hash convolution start at .001; the rest of the model starts at .01. Both follow the existing32-epoch cosine schedule with the same weight decay. Global clipping is disabled. This is a disclosed optimizer-recipe change, intended to reduce abrupt route changes without slowing the continuous GDN/content network. Parameter grouping is verified for completeness and uniqueness, and exact group rates/names are logged. One write/four reads and the original baseline gates remain unchanged.

22. `22_verified_transport`: Run the intended trial07 settings under corrected Python module resolution: no scalar transport decay, isolated learned hash, content-only writes, incidence scaling, and genuinely active full neighboring write gradients. Python safe-path mode prevents the training working directory from shadowing overlaid modules. The GPU validation process and the actual trainer independently verify module paths and SHA256 digests against the mounted intended sources before training. Original unclipped, single-group AdamW .01 recipe; one write/four weighted reads; unchanged strict baseline gates. Earlier trials are flagged because they trained against the frozen router.

Verified completion: trial22 passed all15 gates and finished epoch32 at81.77875% versus GDN69.8621875%. Final/selected intermediate checkpoints, plots, LaTeX and CSV tables, and runtime source audits are archived. See `22_verified_transport/FINAL.md`. The Modal app has been stopped.
