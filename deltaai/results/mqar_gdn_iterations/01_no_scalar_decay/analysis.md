Trial 1 stopped automatically at epoch 4: 42.2528% vs GDN 60.6588%.
Removing scalar erasure improved the prior neighbor-gradient run (27.4681% at epoch4),
but did not restore baseline performance. Accuracy fell from 45.6866% at epoch3,
indicating an unfavorable learning trajectory rather than an absence of all learning.
Next: isolate write-hash surrogate gradients from backbone inputs; keep the
learned causal hash convolution and hash parameters trainable, with identical forward
behavior. This tests interference between routing and shared representation learning.
