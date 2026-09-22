# Revision text: evaluated recurrent models and their decoding costs

Prepared 2026-09-22 from local implementations. This is insertion-ready draft
material, not an edit to the manuscript PDF. Before publication, map each
reported result to its frozen source/configuration. The live wrappers below
establish the current configurations; they do not by themselves certify every
historical checkpoint in the paper.

## Placement and edits

1. In Section 3.4, identify the GDN memory as **boundary-transported additive
   profile memory**. State the post-tabulation learned decoding cost explicitly.
2. Extend Appendix G.1 with the definitions, recurrences, transport, readout and
   pseudocode below. Keep addressing/STE training in a separate paragraph.
3. Label the existing complexity row **hard-routing additive SMat**. State that
   its decoding column is after fixed-prefix tabulation. Add separate learned
   Mamba-2 and learned GDN rows rather than attributing transport to both.
4. Include a recipe table linking each experiment to its memory update, decay,
   hash gradient rule, read count, feature dimensions and frozen source revision.
5. Keep the additive theorem's scope unchanged. The new costs below follow from
   explicit operations, not from transferring the hard-routing theorem to GDN.

## Notation and features

Work per layer and head. Fix the distant/recent boundary n. Let h_t be the layer
input, x_j its hard write profile for j <= n, X the profile set (size N), and C
the B-by-N incidence matrix. A profile/type state has shape r-by-p.

Initialize both local recurrent states and backbone convolution histories to
zero at the start of each half. Routing is causal as described in Appendix G.1.
Write addresses are fixed from the distant token's causal representation. The
backbone features below come from its learned projections and short causal
convolutions. This reset applies to backbone feature convolutions; the separate
write-address convolution only contributes distant-prefix addresses.

For GDN, q_t and k_t are the L2-normalized query/key convolution outputs; v_t is
the value convolution output. With raw scalar gate projections b_t and g_t,

\[
\beta_t=\sigma(b_t),\quad
a_t=\exp[-\exp(A_{\log})\operatorname{softplus}(g_t+\delta)],\quad
A_t=a_t(I-\beta_t k_tk_t^\top).
\]

These are the current `allow_neg_eigval=False` settings. A configuration enabling
negative eigenvalues instead uses beta_t = 2 sigma(b_t); disclose that separately
if such a checkpoint is reported. The local GDN recurrence is

\[
H_t=A_tH_{t-1}+\beta_t k_tv_t^\top,\qquad
z_t^{\rm local}=r^{-1/2}q_t^\top H_t.
\]

For Mamba-2, write k_t=B_t, q_t=C_t, v_t=x_t for the corresponding convolution
outputs, and define Delta_t=softplus(dt_t+dt_bias), with configured clipping if
present, and a_t=exp(-exp(A_log) Delta_t). The local recurrence and direct skip are

\[
H_t=a_tH_{t-1}+\Delta_t k_tv_t^\top,\qquad
z_t^{\rm local}=q_t^\top H_t+D_{\rm skip}v_t.
\]

The memory write gate is a separate projection
\(\omega_t=\sigma(w_{\rm write}^\top h_t+b_{\rm write})\), not Delta_t.

## Mamba-2 profile memory

Initialize all F_x to zero and, for j <= n, define

\[
F_x^{(j)}=a_j^G F_x^{(j-1)}
 +\mathbf1\{x_j=x\}\omega_j k_jv_j^\top.
\]

Here a_j^G=1 in the current recall wrapper; the PG-19 wrapper uses a_j^G=a_j.
Thus inactive profiles stay unchanged when memory decay is off and undergo the
same scalar decay when it is on. There is no profile-specific delta correction.
The decay is implemented through factored weights, without physically visiting
every profile at every step:

\[
F_x^{(n)}=\sum_{j:x_j=x}
\left(\prod_{s=j+1}^{n}a_s^G\right)\omega_j k_jv_j^\top.
\]

Set U_b=sum_x C_bx F_x^(n). For recent token t, use the memory query

\[
\bar q_t=\left(\prod_{s=n+1}^{t}a_s^G\right)q_t.
\]

Both products are one when memory decay is disabled. The identity content
kernel is used: there is no attention denominator based on key mass.

## GDN boundary-transported profile memory

Do **not** describe this implementation as independently applying a routed
delta update only to the selected profile. Define ordered transition products

\[
P_{b:a}=A_bA_{b-1}\cdots A_a,
\quad P_{b:a}=I\text{ if }b<a.
\]

The implemented distant keys and recent queries are

\[
\bar k_j=P_{n:j+1}\beta_j k_j,\qquad
\bar q_t=r^{-1/2}P_{t:n+1}^{\top}q_t.
\]

After computing the transported keys, profile storage is additive:

\[
F_x=\sum_{j\le n}\mathbf1\{x_j=x\}\bar k_jv_j^\top,
\qquad U_b=\sum_x C_{bx}F_x.
\]

For completeness, the mathematically equivalent original-coordinate prefix
recurrence is

\[
F_x^{(j)}=A_jF_x^{(j-1)}
 +\mathbf1\{x_j=x\}\beta_j k_jv_j^\top.
\]

The transition A_j acts on **every** profile in this equivalent recurrence,
including inactive ones; only the new payload injection is profile-routed.
The implementation avoids that all-profile update by transporting keys before
pooling. The stored additive table is assembled after the distant prefix; the
address of each write is nevertheless computed causally and is not reassigned
using later queries. Prefix transport may depend on later positions within the
distant block, all of which precede the recent queries.

This differs from the older routed profile-delta option,
F_x <- F_x + 1{x_j=x} beta_j k_j (v_j^T-k_j^T F_x), which leaves inactive profiles
unchanged. Include that older equation only for results whose frozen recipes
actually use it. Do not use it as the definition of the transport model.

## Learned selection, gating and output processing

With read-feature width w (w equals model width for the dense unreduced reader),

\[
\ell_t=W_R\operatorname{LN}(h_t)+b_R,\quad
I_t=\operatorname{Top4}(\ell_t),\quad
\rho_{tb}=\frac{\exp\ell_{tb}}{\sum_{c\in I_t}\exp\ell_{tc}}
\quad(b\in I_t).
\]

All B scores are evaluated before selection. Use the actual learned projection
before W_R if a reported experiment uses a reduced-rank reader.

\[
m_t=\bar q_t^\top\sum_{b\in I_t}\rho_{tb}U_b,\quad
\lambda_t=\sigma(\alpha+w_\lambda^\top h_t+b_\lambda),\quad
z_t=z_t^{\rm local}+\lambda_t m_t.
\]

For t <= n, z_t=z_t^local. No memory term or attention-denominator normalization
is added there. For both backbones, combine local and memory outputs **before**
the backbone's output gating/normalization and projection. State the order:

- GDN: per-head RMSNorm of z_t, multiply by SiLU of the output-gate projection,
  concatenate heads and apply W_out.
- Current Mamba-2 default (`norm_before_gate=False`): multiply by SiLU of its
  output gate, apply the backbone's configured grouped RMSNorm, then W_out.
  Retain the local direct skip already included above.

The standard layer residual and, for the language models, the feed-forward
block follow the backbone wrapper. State their configuration separately.

## Compact GDN pseudocode

```text
Compute backbone features/gates with convolutions reset at n.
Run local GDN independently on [1,n] and [n+1,T].
Compute distant causal write addresses x[j].

F[:] = 0; W = I
for j = n,...,1:
    transported_key = W @ (beta[j] * k[j])
    F[x[j]] += outer(transported_key, v[j])
    W = W @ A[j]             # rank-one formula, not dense matrix multiplication
U = incidence_aggregate(C, F)
discard F after tabulation

D = I
for t = n+1,...,T:
    D = D @ A[t]             # D = (A[t] ... A[n+1])^T, since each A is symmetric
    query = D @ q[t] / sqrt(r)
    logits = dense_read_scorer(h[t])    # B logits, not four
    ids = top4(logits)
    weights = softmax(logits[ids])
    memory = query^T @ weighted_sum(U[ids], weights)
    z = local_output[t] + sigmoid(read_gate(h[t])) * memory
    emit backbone_output_processing(z, h[t])
```

For Mamba-2, replace matrix transport by the scalar products above and replace
beta by the independent sigmoid write gate. With decay disabled, both transport
factors are one. Use the Mamba local recurrence and output normalization order.

## Complexity table and assumptions

Costs are per head/layer, with fixed read count K=4. Let N be the profile count,
B the type count, w the scorer input width, and A_C the work to aggregate
profiles into summaries. Standard feature projections, short convolutions,
write hashing, residuals, FFNs and the vocabulary head are omitted from these
mixing-core costs and must be stated as common additional costs. These costs
are arithmetic work, not parallel training depth or measured throughput.

| Model | Full-sequence mixing work | Per recent decoded token after fixed-prefix tabulation | Persistent dynamic cache (words) |
|---|---|---|---|
| Hard-routing additive SMat | O((T+T^(2-3/d))rp) | O(rp) | O(Brp) |
| Learned four-read Mamba-2 SMat | O(Trp + A_C + TBw) | O(rp + Bw) | O(Brp + rp) |
| Learned four-read transported GDN SMat | O(T(rp+r²) + A_C + TBw) | O(rp + r² + Bw) | O(Brp + rp + r²) |

The GDN r² term accounts for the boundary-to-query transport matrix. A rank-one
transition W A_t = a_t[W-beta_t(W k_t)k_t^T] costs O(r²); it is not a generic
O(r³) matrix multiplication. Local GDN and payload reads cost O(rp).
Top4 over B logits costs O(B) with fixed K, absorbed in O(Bw).

For the practical dense incidence contraction, A_C=O(BNrp). Under the field-size
choice in the paper, N,B=Theta(T^(1-1/d)), hence A_C=O(T^(2-2/d)rp).
If the sparse incidence algorithm is used, A_C=O(T^(2-3/d)rp). Label which
implementation the reported runs actually use; do not silently substitute the
sparse bound for a dense einsum/matmul.

With all feature dimensions fixed, the learned rows have Theta(T^(2-1/d))
sequence-wide scoring work and Theta(T^(1-1/d)) per-token scoring work. At d=3,
these are Theta(T^(5/3)) and Theta(T^(2/3)).

Dynamic cache counts exclude parameters and fixed geometry. In particular, the
learned scorer stores O(Bw) parameters, and a materialized dense incidence matrix
uses O(BN) storage. During prefix construction both F and U may coexist; after
tabulation F can be discarded. Separately report actual convolution caches,
static buffers, dtype, and measured GPU allocations.

Suggested Section 3.4 sentence:

> After fixed-prefix summary tabulation, the learned four-read selector still
> evaluates all B type scores. Its routing cost is Theta(Bw) per decoded token,
> in addition to the local recurrent update and selected-summary contractions;
> transported GDN also maintains an r-by-r transition state. Thus at fixed model
> dimensions the learned variant costs Theta(T^(1-1/d)) per decoded token,
> whereas the hard-routing additive construction has O(rp) post-tabulation cost.

## Source checks and remaining manuscript mapping

- `deltaai/zoo_gdn_transport.py`: transport-additive default, local resets,
  read gate, r^(-1/2) query scaling and output processing.
- `deltaai/smat_gdn_transport.py`: ordered transition products and shifted
  reverse prefix scan, with beta included exactly once.
- `deltaai/zoo_mamba_four_reads.py`: recall wrapper, independent write gate,
  memory decay disabled.
- `deltaai/lm/mamba_smat_scale.py`: PG-19 wrapper, memory decay enabled.
- `deltaai/zoo_smat_mixer.py`: Mamba local recurrence, memory weighting,
  scalar decay factoring and gated RMSNorm output.
- `deltaai/lm/pg19_recurrent.py`: cached learned top4 selection, recurrent GDN
  transport state, and Mamba scalar decay during decoding.

Audit the paper's hash-gradient paragraph too: the latest local PDF describes
the retained-route one-sided STE, while the current GDN configuration enables
neighbor-route surrogate gradients. Specify the rule from each run's frozen
source; forward one-write behavior alone does not identify the backward rule.

Do not treat the older `smat/lifting_framework.md` specialization table as the
current GDN definition: it describes the earlier soft profile-delta model.
