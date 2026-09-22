Rejected at epoch4: 43.7865625% vs60.65875%.
Shared address-derived reads and cross-length hash sharing together regress
from trial03's56.64%. The run learns monotonically, but is too slow for the
required epoch4 criterion. This does not isolate which of the two changes
is responsible. Next trial retains hash sharing and restores the previous
categorical reader, isolating the cross-length hypothesis.

Saved-weight analysis: first-layer coordinate cosine similarities are
0.999799 and0.998405, with identical hash parameters across lengths.
Thus the shared-address option collapsed its two coordinates almost onto
one direction. This is an observed parameter degeneracy, not proof that
an orthogonality constraint alone would restore accuracy.
