# Soft routing with controlled support complexity

Research note, 2026-09-15. This proposes and analyzes a router variant; it does not change the model implementation or report a trained result for that variant.

## Recommendation

Keep the successful soft-write/soft-read GDN model as the reference. Its four-by-four routing at d=3 already admits a meaningful support-VC analysis. The number 16 in its pairwise expansion is an upper-bound bookkeeping device, not its actual VC dimension or number of query reads.

The first alternative worth testing is **simplex interpolation on both sides**. It replaces 2^(d-1) neighboring grid vertices with at most d vertices. This retains continuous forward routing and ordinary derivatives almost everywhere. For d=2,3,4 it needs at most four routes per side. It is not a constant-route construction for arbitrarily large d.

Do not replace the working model until this variant passes a matched training comparison. Neither the existing two-layer results nor the checks below establish stability at 15 layers.

## 1. Existing mechanism and evidence

`deltaai/content_addr.py`, `ContentAssign._cells`, implements multilinear interpolation in D=d-1 hash coordinates. The persistent-soft configuration uses this interpolation in the forward pass for both keys and queries, with `anneal=0`. It enumerates all 2^D corners; this is not a learned top-k ranking over all profiles.

The sparse hyperplane branch chooses one direction and maps each query vertex to the hyperplane through it. Some vertices can map to the same hyperplane. Therefore its number of distinct summary reads can be smaller than its number of interpolation vertices.

The verified soft-routing MQAR accuracies are 78.05%, 94.29%, and 96.85% at widths 16,32,64 for the completed seed-123, 32-epoch runs. See `deltaai/results/mqar_gdn_soft_routes/quoted_heads_s123/reevaluation.json`. These support preserving the jointly soft design. They do not isolate the contribution of soft writes from soft reads.

## 2. A sharper support-VC upper bound

Let C be the point-hyperplane incidence matrix over F_q^D. Let R and S be nonnegative routing matrices with rows summing to one, with at most K_r query types per row of R and K_w profiles per row of S. Define

\[
G=RCS^\top,\qquad B_{ij}=\mathbf 1[G_{ij}>0].
\]

All support statements below concern this structural routing relation. For delta memory, G is not a claim that the output's numerical value coefficients factor as RCS^T: subsequent state updates transform earlier writes.

For a fixed sequence and its fixed key routing, write

\[
X_j=\{x:S_{jx}>0\},\qquad |X_j|\le K_w.
\]

Each key is now a small **bag of profiles**. A query selecting hyperplanes H_1,...,H_k, with k<=K_r, reaches key j exactly when

\[
X_j\cap(H_1\cup\cdots\cup H_k)\ne\varnothing.
\]

This is the OR rule used in multiple-instance learning. The relevant established connection is Sabato and Tishby, *Multi-Instance Learning with Any Hypothesis Class*, JMLR 2012, especially Lemma 5 and Theorem 6:

https://jmlr.csail.mit.edu/papers/volume13/sabato12a/sabato12a.pdf

Their setting studies classification of bags; the application here is a bound on routing row supports, not a generalization theorem for the neural network.

### Direct counting argument

The family of proper affine hyperplanes has VC dimension v=D. Consider t selected key columns. Their bags contain at most K_w t distinct profiles. By Sauer's lemma, a single hyperplane has at most

\[
\sum_{\ell=0}^{\min(v,K_wt)}\binom{K_wt}{\ell}
\]

traces on these profiles. Choosing up to K_r hyperplanes gives at most

\[
\Pi_B(t)\le
\left[\sum_{\ell=0}^{\min(v,K_wt)}\binom{K_wt}{\ell}\right]^{K_r}
\]

different key-support patterns. Padding a nonempty selection with repeated hyperplanes covers the case of fewer than K_r choices. Requiring all selected hyperplanes to share a direction only restricts this family, so the current implementation satisfies this bound too.

For t>=v, shattering therefore requires

\[
2^t\le\left(\frac{eK_wt}{v}\right)^{vK_r}.
\]

Substituting u=t/(vK_r) gives 2^u <= e K_w K_r u. Consequently,

\[
\boxed{\operatorname{VC}(B)
=O\!\left(vK_r\log(2K_wK_r)\right).}
\]

This is a direct adaptation of the trace-counting method, not a theorem specifically about SMat quoted from the reference. It improves the earlier generic K_w K_r-component union bound by using the fact that one query hyperplane tests all write destinations together.

In particular, the write budget enters this upper bound logarithmically. This does not make writes computationally free, and it does not identify the actual VC dimension of a trained mask.

### Adding the two causal blocks

For the full structural mask

\[
M=\begin{pmatrix}L_n&0\\ B&L_m\end{pmatrix},
\]

one always has VC(M)<=VC(B)+2:

- A shattered set cannot contain two recent columns, because all row traces on recent columns are prefixes.
- If it contains one recent column, label that column one. The recent rows must then shatter all the selected distant columns, giving at most VC(B)+1 columns in total.
- If all t selected columns are distant, fix the earliest selected column to zero and the latest to one. No distant prefix row realizes that pattern. Thus the recent rows must shatter the remaining t-2 columns, giving t<=VC(B)+2.

Hence

\[
\boxed{\operatorname{VC}(M)
=O\!\left((d-1)K_r\log(2K_wK_r)\right)+2.}
\]

This is a uniform bound for each realized matrix with fixed key bags. It is not the VC dimension of the full parameterized network over raw inputs.

With K_r,K_w bounded independently of d and T, this is O(d). With simplex routing K_r,K_w<=d, it is O(d^2 log(2d)). The generic bound for the current K=2^(d-1) router becomes O(d^2 2^d), still independent of T for fixed d. These are upper bounds, not claims that the respective orders are attained.

## 3. Simplex interpolation: a minimal change to the soft router

Let z be the continuous hash coordinate and write z=ell+f in an interior grid cell, with ell integral and 0<=f_l<=1. Sort its fractions:

\[
f_{\pi_1}\ge\cdots\ge f_{\pi_D}.
\]

Select the chain of D+1 vertices

\[
x_0=\ell,\qquad
x_a=\ell+\sum_{b=1}^{a}e_{\pi_b},\quad 1\le a\le D,
\]

and weights

\[
\alpha_0=1-f_{\pi_1},\quad
\alpha_a=f_{\pi_a}-f_{\pi_{a+1}}\ (1\le a<D),\quad
\alpha_D=f_{\pi_D}.
\]

All weights are nonnegative and sum to one. At the upper grid boundary, clamp vertices and merge duplicate destinations as the current router already does. Coordinate reproduction there applies to the clamped coordinate.

For d=3, D=2. If f_1>=f_2, the routes are

\[
(0,0),(1,0),(1,1),\qquad
\alpha=(1-f_1,f_1-f_2,f_2),
\]

translated by ell. If the order reverses, use the other triangle. At f_1=f_2 the differing intermediate vertex has zero weight, so the dense routing distribution is continuous across the change of triangle.

This standard construction is described in Scott Davies, *Multidimensional Triangulation and Interpolation for Reinforcement Learning*, NeurIPS 1996, Section 1.2:

https://proceedings.neurips.cc/paper/1996/file/310ce61c90f3a46e340ee8257bc70e93-Paper.pdf

It is also implemented by TensorFlow Lattice:

https://www.tensorflow.org/lattice/api_docs/python/tfl/lattice_lib/evaluate_with_simplex_interpolation

### What it preserves

- **Soft forward writes and reads.** No annealing to one-hot routing is required.
- **Ordinary coordinate derivatives almost everywhere.** Within a nondegenerate simplex the weight Jacobian has rank D.
- **Coordinate marginals.** The total weight on vertices incrementing coordinate l is exactly f_l. Thus the existing per-coordinate occupancy KL remains the correct marginal statistic. Joint profile occupancies can change.
- **The finite geometry and state budget.** C, profile identities, and the O(T^(1-1/d)) number of cached states are unchanged.
- **A route-level hard special case.** At vertices, weights become one-hot. Recovering the draft's exact-d witness construction additionally requires realizing the necessary vertex/type assignments with the chosen encoder or planting mechanism; it does not follow just from having interpolation formulas.

| d | Current maximum vertices per side | Simplex maximum vertices per side | Simplex pair components |
|---|---:|---:|---:|
| 2 | 2 | 2 | 4 |
| 3 | 4 | 3 | 9 |
| 4 | 8 | 4 | 16 |
| 5 | 16 | 5 | 25 |

These are maximum counts before merging duplicates and zero weights. Pair components count terms in an expansion, not distinct summary reads or VC dimension. Even the first three rows do not use exactly the same number of active routes; they share a maximum budget of four.

### Delta-memory compatibility

The current update for one routed profile is

\[
F_x\leftarrow F_x+\beta_j S_{jx}k_j(v_j^\top-k_j^\top F_x).
\]

It directly accepts simplex weights. A route vanishing at a simplex boundary becomes an identity update, so the per-profile recurrence remains continuous in the cell coordinates for a fixed finite sequence. Duplicate routes must be merged before the delta update, as in `smat_delta_pool.py`.

For a unit-norm key and 0<=beta*S<=1, the homogeneous state map I-beta*S*k*k^T has operator norm at most one. Simplex weights preserve this condition. This concerns the memory update alone, not stability of an entire deep network.

The query direction selector remains the existing hard choice. Continuity claims here concern cell interpolation with direction held fixed, not every routing operation in the model.

### What changes and could hurt training

The joint distribution over corners changes. Multilinear interpolation behaves like independent randomized rounding of each coordinate; simplex interpolation couples those rounding choices through one shared uniform variable, integrated out exactly. It preserves marginals but changes which coordinates move together.

Simplex interpolation also introduces derivative changes along boundaries inside a grid cell. At its center it can put all mass on two opposite vertices. Therefore preserving gradients does not imply reproducing the successful router's optimization behavior or accuracy.

## 4. Why an arbitrary constant K has a real tradeoff

Suppose a local router retains coordinate reproduction,

\[
z=\sum_{a=1}^K w_a(z)x_a,\qquad \sum_a w_a(z)=1,
\]

with locally fixed discrete vertices x_a. Differentiating yields I_D=V J_w, while 1^T J_w=0, so

\[
D\le\operatorname{rank}(J_w)\le K-1.
\]

Thus K>=D+1=d. Simplex interpolation meets this lower bound.

This is a limitation on full-dimensional, coordinate-preserving local interpolation. It is not an impossibility theorem for all trainable fixed-K routers. A fixed-K router for unbounded d must give up that property or use another gradient mechanism. A locally fixed support with normalized weights has at most K-1 independent weight variations, even if every input coordinate receives some gradient.

Simply keeping the four largest multilinear weights can also make the routing distribution discontinuous: in three hash dimensions, approach (.5,.5,.5) by changing only the first coordinate. On opposite sides, top-4 selects opposite four-vertex faces, and the normalized distributions have L1 distance 2 even as the coordinate perturbation tends to zero. Simplex interpolation avoids this discontinuity because departing vertices lose their weight at the boundary.

## 5. Complexity and the next experiment

Generating simplex candidates takes O(D log D) work for sorting, with O(D) vertices; no scan over all N0 profiles is needed. Write-event and selected-summary-read counts become O(T d) rather than O(T 2^(d-1)), ignoring state width and other layer work. For fixed d, this preserves the additive construction's asymptotic dependence on T, with the appropriate routing factors exposed.

This does not transfer the additive algorithm's complete theorem to the GDN implementation. The current sparse branch still scores all directions, materializes a dense incidence tensor, uses a dense profile-to-type einsum, and sorts profile write events. In particular, the number of scored directions grows with q for D>1. A constant number of selected reads does not make the current full decoding implementation independent of T. Those costs require separate profiling and implementation work.

The first matched experiment should change only the cell interpolator at d=3, on both reads and writes. Keep the successful delta updates, q, state shapes, key source, normalization, beta, read gate, loss, and training schedule. Compare the existing four-corner model with the three-vertex simplex model across several seeds. Record accuracy, throughput, gradient norms, route entropy, occupancy, and the realized density of B. This is a test of whether fewer soft routes preserve performance, not an experiment isolating VC dimension as the cause of accuracy.

If that comparison succeeds, test d=4 (eight versus four routes), then increase depth with otherwise fixed configuration. Keep the original router if the performance loss outweighs the savings.

## 6. Verification completed

A temporary CPU/PyTorch prototype checked the proposed formulas, separately from production FLA kernels:

- Nonnegative weights, unit row sums, coordinate reproduction, and full local weight-Jacobian rank for d=2,3,4,5.
- Coordinate-moment agreement with multilinear interpolation to at most 1.34e-15 in double precision.
- Continuity probes at simplex faces, grid faces, and the upper clamp: routing L1 changes at most 4.01e-7 for coordinate perturbations of order 1e-7.
- Finite-difference/autograd agreement for routing and for a small chronological delta-memory recurrence followed by a fixed-direction hyperplane read. Every tested coordinate had a nonzero output gradient.
- The top-4 truncation counterexample above (L1 jump 2, versus approximately 4e-7 for the continuous interpolators).

These verify local mathematical and implementation properties of the prototype. No simplex SMat model was trained, no GPU kernel performance was measured, and no production model code was changed.

## 7. A fully soft subset-routing witness for the current router

Added 2026-09-16. This witness uses the existing four-corner multilinear interpolation, not the proposed simplex interpolation. It establishes a property of the allowed routing configurations, not the measured behavior of a trained checkpoint.

Take d=3, D=2, q=7, and three distant keys with continuous grid coordinates

\[
z_A=(1.5,1.5),\quad z_B=(3.5,1.5),\quad z_C=(1.5,3.5).
\]

Each key writes weight 1/4 to each of the four corners of its grid cell. For every query below, both coordinates also have fractional part 1/2, so its four cell routes have weight 1/4. Its chosen finite-field direction a maps those cells y to hyperplanes H_{a,a^T y}. All arithmetic defining incidence is modulo 7. Repeated hyperplane destinations can be merged.

Keep the three key assignments fixed and choose the following queries:

| Desired subset | Query grid coordinate | Direction a | (G_iA, G_iB, G_iC) |
|---|---|---|---|
| empty | (0.5,5.5) | (0,1) | (0,0,0) |
| A | (0.5,0.5) | (1,1) | (1/16,0,0) |
| B | (3.5,0.5) | (1,0) | (0,1/2,0) |
| C | (0.5,3.5) | (0,1) | (0,0,1/2) |
| A,B | (0.5,0.5) | (0,1) | (1/4,1/4,0) |
| A,C | (0.5,0.5) | (1,0) | (1/4,0,1/4) |
| B,C | (0.5,5.5) | (1,1) | (0,1/4,1/4) |
| A,B,C | (0.5,2.5) | (0,1) | (1/4,1/4,1/4) |

Here G=RCS^T is the structural routing coefficient. Each entry was verified by enumerating the 16 query-corner/key-corner pairs and dividing the number of incidences by 16. Therefore the structural cross-block support shatters these three keys. Every nonzero entry is at least 1/16 in this example, so the same subset patterns also survive any uniform threshold strictly between zero and 1/16.

All four cell weights on both sides are strictly positive. Small coordinate perturbations that stay within the same grid cells preserve the support patterns, provided the query directions stay fixed. This is not a shattering witness obtained solely by taking a one-hot limit. It does retain the current implementation's discrete direction choice.

This demonstrates possible cross-block VC at least 3 for this d=3 soft-routing construction; the original hard point-line incidence cross-block has VC at most 2. Thus soft support need not obey the hard construction's exact VC value. The full temporal mask can add further expressivity; the cross-block witness alone is not an exact VC calculation for the full mask.

The witness specifies valid grid coordinates and directions at the router interface. Realizing it through a particular trained encoder, and successfully recovering requested values from its profile states, are separate questions. For GDN, the numbers in the table describe routing overlap, not the numerical coefficients after delta-state updates. The existing MQAR results do not by themselves certify all-subsets routing in a trained checkpoint.
