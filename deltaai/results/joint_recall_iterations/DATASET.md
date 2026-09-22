# Joint context–key recall

Each example contains a randomly generated table of bindings from a context–key
pair to a value. Every context uses the same set of keys, while each pair receives
an independently sampled value. A key alone therefore does not uniquely identify
a record when more than one context is present.

Information records are serialized as `(context, key, value)` in random order.
After a separator, independently shuffled queries ask for the value associated
with every stored pair. Queries have the form `(context, key, PAD)`; answer slots
are masked, and earlier answers are never supplied as input. The prediction at
the query's key position is supervised with the corresponding value.

Context identifiers are sampled without replacement from 32 context tokens, and
the shared key set is sampled without replacement from 512 key tokens. Values are
sampled independently and uniformly from 16 value tokens for every binding and
every example. Including three special tokens, the vocabulary has 563 tokens.
Uniform guessing achieves 6.25% per-query accuracy.

| Contexts | Keys per context | Bindings | Sequence length | Training examples |
|---:|---:|---:|---:|---:|
| 1 | 4 | 4 | 64 | 36,000 |
| 2 | 8 | 16 | 100 | 36,000 |
| 8 | 16 | 128 | 772 | 36,000 |
| 16 | 16 | 256 | 1,540 | 36,000 |
| 32 | 16 | 512 | 3,076 | 36,000 |

The training split contains 180,000 fixed examples, evenly divided across the five
sizes and interleaved during training. Development validation and test splits
contain 2,000 and 4,000 examples per size, respectively. Tables and query orders are
generated independently across examples and splits. The splits share the same
identifier distribution; the task does not withhold particular context–key
combinations or evaluate out-of-distribution identifiers.

The main metric is per-query accuracy as a function of the number of stored
bindings. Also report exact-table accuracy, which requires every query in an
example to be correct, and macro accuracy averaged equally over the five sizes.
The one-context cell provides an ordinary key-recall reference; the other cells
require disambiguation using context and key jointly.
