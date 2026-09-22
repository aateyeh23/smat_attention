Learned causal key construction for SMAT

The memory query is normalize(W x_t); the write key is normalize(W CausalConv4(x)_t). The same learned causal context feeds write routing. The depthwise convolution begins as a uniform four-position average and learns freely; there is no specified key offset, token parity, or key/value identity access. GDN retains its native independent projections. The two-layer model adds 2 * width * heads * 16 projection parameters over the base-key SMAT variant. All widths use seed 123 and 32 epochs of 707 batches, with 1/2/2 GDN heads.

Fixed preceding-token variants are diagnostic ablations and are excluded from the main goal's winning results. This variant retains SMAT's existing generic routing geometry and chunked delta memory.
