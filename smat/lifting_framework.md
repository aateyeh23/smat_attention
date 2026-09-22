# Lifting recurrent sequence models into SMat

## Scope

This is a proposed mathematical formulation of the existing SMat architecture.
It unifies additive Mamba-2 memory and the current soft-routed GDN delta memory.
It does not assert that all specializations inherit the hard-mask VC theorem or
the additive chunkwise attention theorem.

Equations describe one head, conditional on the projected features, gates, and
routing assignments of a forward pass. Input projections and causal convolutions
produce those features; the backbone's output normalization and projection are
applied after combining the local and memory outputs. Mamba's direct skip term
is retained separately. The memory and local branches may use different key and
query features.

## 1. Base model: a recurrent memory update and a readout

Let a base model maintain a matrix state H_t in R^{r x p}. A family covering the
recurrent cores of Mamba-2 and Gated DeltaNet is

\[
H_t=A_t H_{t-1}+b_t k_t v_t^\top,
\qquad z_t=q_t^\top H_t,
\]

where

\[
A_t=a_t\bigl(I-\eta b_t k_tk_t^\top\bigr).
\]

The scalar eta is a notation for the update choice, not an existing trainable
parameter in our code.

| Base recurrent core | eta | a_t | b_t | Features |
|---|---:|---|---|---|
| Mamba-2 | 0 | scalar discretized state decay | discretized write strength, dt | q_t = C_t, k_t = B_t, v_t = x_t |
| Gated DeltaNet | 1 | scalar forget gate | beta_t | normalized projected keys and queries |

For GDN this form predicts from the decayed previous state and then writes the
prediction residual. An ordinary additive memory is the eta = 0 case. The
framework permits separate local and memory choices of these parameters.

## 2. Geometry: write assignments and incidence-based readout

Fix a distant/recent boundary n in a sequence of length N. Let X be the profile
set and H the query-type set, with binary incidence matrix

\[
C_{hx}=\mathbf 1\{x\in H_h\}.
\]

For distant positions j <= n, let S_{jx} >= 0 be the write assignment to profile
x. For recent positions i > n, let R_{ih} >= 0 be the query-type assignment.
Recent indices on R are identified with their original sequence positions.
Define

\[
r_{ix}=\sum_h R_{ih}C_{hx},\qquad G=RCS^\top.
\]

Hard assignments select one profile/type. Our sparse soft assignments select
neighboring cells, with at most K = 2^{d-1} cells per token for geometric
dimension d-1. Query cells induce hyperplanes through a selected direction;
coincident types are understood to have their weights summed.

These objects describe where a token writes and which profile memories a query
can read. They do not prescribe the memory update law.

## 3. The SMat lift

### Local branch

Run the base model separately on positions 1,...,n and n+1,...,N, resetting its
local recurrent state at the boundary. Our reset implementations also restart
the backbone's short convolution. Denote the resulting local outputs by
z_i^{local}.

### Profile memories

Introduce a state F_x for each profile, initialized to zero. For each distant
token j, use memory features bar{k}_j, bar{v}_j and memory write strength
bar{b}_j, and update

\[
\boxed{
F_x^{(j)}=A_{jx}^{G}F_x^{(j-1)}
 +\bar b_jS_{jx}\bar k_j\bar v_j^\top,
\qquad
A_{jx}^{G}=\bar a_j
 \bigl(I-\eta_G\bar b_jS_{jx}\bar k_j\bar k_j^\top\bigr).
}
\]

When bar{a}_j = 1, this is equivalently

\[
F_x\leftarrow F_x+
\bar b_jS_{jx}\bar k_j
\left(\bar v_j^\top-\eta_G\bar k_j^\top F_x\right).
\]

The write strength bar{b}_j is separate from the routing assignment S_{jx}.
It may reuse the backbone's write strength or come from an independent gate.
For eta_G = 1, their product controls both writing and overwriting.

### Incidence aggregation and readout

After processing the distant block, construct

\[
U_h=\sum_x C_{hx}F_x^{(n)}.
\]

For recent query i, combine the memories and the local output:

\[
\boxed{
z_i=z_i^{local}
 +\lambda_i\bar q_i^\top\sum_h R_{ih}U_h,
\qquad i>n.
}
\]

For i <= n, use z_i = z_i^{local}. Apply the backbone's output processing after
this combination. The read gate lambda_i, memory query bar{q}_i, write strength
bar{b}_j, and assignment S_{jx} are distinct parts of the definition. A
continuation of a common scalar memory decay into the recent block can be
included as an additional scalar multiplier on the memory readout.

The same construction also permits an unreset backbone plus additional memory,
but that is a hybrid variant: its local branch retains its original cross-block
path, so its structural mask is not the two-triangle mask below. Setting
lambda = 0 recovers the chosen local branch; in the reset variant it does not
recover an unreset base model.

## 4. Current specializations

| Choice | SMat + Mamba-2, checked reset configuration | Current soft SMat + GDN |
|---|---|---|
| Local branch | Mamba-2 within each half | GDN within each half |
| Memory update eta_G | 0: additive | 1: delta |
| Memory decay | none in checked configuration; optional shared scalar decay exists | none |
| Memory write strength | independent sigmoid gate replacing local dt | backbone beta, reused |
| Memory keys/queries | backbone B/C for the identity kernel | separate shared memory projection on causal write context and query context |
| Routing | content assignments; checked recipe anneals to hard | sparse neighboring-cell mixture remains soft |
| Readout | incidence aggregation, content contraction, learned read gate | same organization |

Hard/soft routing, additive/delta updates, and shared/independent memory features
are independent architectural choices. The implementations do not currently
expose every combination through every configuration.

## 5. Exact sequence-operator interpretation

For fixed features and routing, define the ordered transition product

\[
T_x(n,j)=A_{nx}^{G}A_{n-1,x}^{G}\cdots A_{j+1,x}^{G},
\qquad T_x(n,n)=I.
\]

Unrolling each profile state gives the exact memory coefficient

\[
\boxed{
(P_G)_{ij}=\lambda_i\sum_x
r_{ix}\bar b_jS_{jx}
\bar q_i^\top T_x(n,j)\bar k_j,
\qquad O_G=P_G\bar V_{dist}.
}
\]

This equation is the common operator form for both update rules. It is linear
in the value matrix conditional on the other features and routing; the complete
network need not be linear in its inputs.

### Additive specialization

For eta_G = 0 with a profile-independent scalar decay, T_x(n,j) = d_j I where
d_j = product_{t=j+1}^n bar{a}_t. Consequently,

\[
P_G=(\bar Q\bar K^\top)\odot
\left[\operatorname{diag}(\lambda)G
\operatorname{diag}(\bar b\odot d)\right].
\]

If local and memory features and values agree, the reset sequence-mixing core
has one shared content factor and a block-structured weighting matrix:

\[
P=QK^\top\odot
\begin{pmatrix}
W_{local,n}&0\\
\operatorname{diag}(\lambda)G\operatorname{diag}(\bar b\odot d)
 &W_{local,N-n}
\end{pmatrix}.
\]

Independent features require separate branch kernels instead. The formula is
for the unnormalized additive core; any finite-feature attention normalizer
must be included separately.

### Delta specialization

For eta_G = 1, T_x(n,j) depends on the profile's routed update history. Thus P_G
is a sum of profile-dependent interactions. In general it is not an unchanged
global GDN operator multiplied elementwise by G. The routing influences the
state transitions as well as the final readout.

## 6. Structural guarantees and their scope

For positive write/read gates, define the structural value-routing mask

\[
M_{route}=\begin{pmatrix}
L_n&0\\
\mathbf 1\{G>0\}&L_{N-n}
\end{pmatrix}.
\]

- **Causality:** both update choices preserve causality with the fixed boundary.
- **Permitted value paths:** the conditional memory coefficient matrix satisfies
  supp(P_G) subseteq supp(G). Equality requires additional noncancellation
  assumptions and need not hold for signed features or delta updates. This is
  not a claim about all input dependencies through feature construction.
- **Hard-mask VC:** with the paper's prescribed hard assignments and witnesses,
  the routing construction has VC dimension d, independently of which memory
  update law is placed behind it. This characterizes permitted access, not
  exact nonzero coefficients or the learned model's total capacity.
- **Soft routing:** its support or weighted pseudodimension requires separate
  analysis; neither is automatically d.
- **Structured computation:** with no memory decay and K routes per token, the
  memory admits O(K n r p) sequential rank-one update work, followed by sparse
  incidence aggregation and O(K r p) state readout work per recent query.
  A common scalar decay can be handled by factoring it out. These statements
  exclude routing-network evaluation, geometry construction, and packing/sorting.
  They do not establish the parallel training cost of a particular implementation.
- **Theorems:** the existing additive chunkwise attention theorem applies to its
  specified additive kernel and mask. A delta implementation's parallel forward
  and backward costs require their own analysis. Current dense incidence
  contractions must not be identified with sparse incidence costs without
  checking their implementation.

The shared geometry uses the same number of profile/type state slots for either
update rule. Under the paper's field-size choice, this is O(N^{1-1/d}) slots,
each holding r p scalars, for fixed geometric d >= 2. This counts persistent
memory organization, not total training activations.

## Suggested paper summary

> SMat lifts a recurrent sequence model by retaining its local computation and
> introducing a collection of profile-indexed memories. Tokens update these
> memories through sparse hard or soft assignments, and queries access them
> through a shared finite-field incidence structure. The memory update is a
> backbone-dependent choice: additive updates yield a factorized masked-attention
> operator, whereas delta-rule updates yield profile-specific recurrent
> operators. This separates routing geometry from memory dynamics and accommodates
> both SMat–Mamba-2 and SMat–Gated DeltaNet within one construction.

## Implementation references

- `deltaai/zoo_smat_mixer.py`: shared B/C memory features, independent/residual
  memory writes, optional scalar memory decay, and additive readout.
- `deltaai/zoo_mamba_reset_configs.py`: checked Mamba reset configuration.
- `deltaai/zoo_smat_gdn.py`: reset GDN, causal memory features, soft routing,
  reused beta gates, and final branch combination.
- `deltaai/content_addr.py`: sparse write/read assignments and incidence readout.
- `deltaai/smat_delta_pool.py`: per-profile delta updates with beta times the
  routed assignment weight.
