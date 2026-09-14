# Width-32 GDN versus GDN+SMAT: matched epoch-3 analysis

Both use interleaved batches, seed123, width32, two layers, four heads of key/value dimension16, lr0.01. GDN scores76.9481%; SMAT d3 scores49.0228%. Exact cell scores are in width32_epoch3_comparison.json.

Confirmed implementation differences:
- Plain GDN carries its recurrent state and short convolutions through the full sequence. The hybrid folds the sequence into two independent halves before the convolution and gated-delta scan.
- SMAT memory writes sigmoid(beta)*k*v, without the GDN delta prediction-correction term or temporal decay. This is not a GDN update inside each memory profile.
- Retrieving original first-half information in the second half therefore requires learned content routing and the additive SMAT memory. The layer-input key hash also uses an extra learned convolution; it must align the value position with the query key.
- Hash parameters for all three lengths are registered before optimizer construction. GPU validation previously established native-GDN equivalence of d1, boundary isolation without G, finite gradients and nonzero G contribution.
- All queries in the16-pair/length64,32-pair/length128 and64-pair/length256 validation cells occur in the second half: the key-value prefix fills the first half. These are especially direct tests of replacement memory. The16-pair gap is93.0125% versus27.51875%.

Interpretation: the architecture removes a strong existing retrieval path and substitutes a harder-to-learn sparse additive path. Routing/alignment instability and interference in additive memory are plausible contributors; their individual causal effects are not measured here. Interleaving alone does not fix the gap. Both models are fully hard-routed by epoch2 where applicable, so epoch3 is not the instant of a soft-to-hard switch. The earlier soft-hash inference probe concerned the fixed-order checkpoint, not this interleaved comparison.

Limits: one seed, three completed hybrid epochs, different parameter sets imply that a common seed does not ensure identical common-layer initialization. Earlier epoch3 baseline weights were overwritten; the current baseline checkpoint is later. This analysis does not establish final converged ranking or attribute all the gap to the boundary alone.

Most informative next architectural control: preserve the full-sequence GDN path and add SMAT as a residual initialized to zero, with identical common parameters. Then separately test GDN-style delta updates inside profiles if desired. No new training configurations or checkpoints were changed during this analysis.
