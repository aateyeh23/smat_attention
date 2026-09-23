# Mamba + SMat MKAR confirmation runs

The final `mamba2-d4-normalized-route-stop` model uses seed 123 for architecture
selection and seeds 456, 789, 2026, and 2027 for independent confirmation.
Each seed has separate k=1, 2, and 3 models trained for 12,000 updates.

The model uses shared memory query/key features, boundary transport, read
normalization, and writer-feature gradient isolation. The local Mamba
recurrence is unchanged. The implementation matches the frozen confirmation
source. Protocols, source hashes, validation records, run recipes, metrics,
final results, and the five-seed d=4 summary document the experiment.

The historical training and preparation scripts require the complete local
source snapshots and parent campaigns named in their provenance records.
Source manifests describe those complete snapshots. Checkpoints and full
snapshots are local artifacts; this record is not a standalone reproduction
bundle. Selection-seed and independent-confirmation results should be reported
separately. These results alone do not establish a routing-specific advantage.
