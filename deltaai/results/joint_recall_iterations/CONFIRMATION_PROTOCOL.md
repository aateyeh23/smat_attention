# Selection and independent confirmation

This protocol is recorded before generating or evaluating the fresh confirmation
tables. The current candidate dataset is explicit context–key recall with mixed
table sizes. Its training split has 60,000 examples at each of C1/K4, C2/K8, and
C8/K16: 180,000 examples total. All four compared models must use this identical
cached dataset. The earlier large-only training results are a separate experiment
and cannot supply a winner for this comparison.

## Development selection

1. Complete the scheduled native and SMat training candidates on this dataset.
   Every completed candidate receives 32 epochs, 22,560 optimizer steps, and
   5.76 million example presentations. There is no accuracy-based early stopping.
2. Select one native and one SMat final checkpoint for each backbone using
   final-epoch validation macro accuracy. Retain every trial and report all cells.
   Validation-best checkpoints are secondary development results, not candidates
   for the primary final-checkpoint comparison.
3. A candidate comparison must have positive SMat gains in both macro accuracy
   and the largest-table cell for both backbones before confirmation is launched.
   Check the other completed native candidates too; if one has stronger large-cell
   accuracy than the selected native model, disclose it and include it in the
   confirmation comparison before claiming a high-load advantage.
4. Record the selected run paths, dataset-manifest hash, frozen source, and
   checkpoint SHA256 hashes in `selection.json` before fresh data generation.
   Any memory-option change must be identified explicitly; a no-decay GDN
   ablation cannot be described as the original decay-enabled model.

## Fresh evaluation

Generate 10,000 independent tables per cell using split code 9001. This code is
distinct from the training, development-validation, and development-test codes.
All selected models answer exactly the same fresh tables, with answers masked
in their inputs. The split follows the same identifier distribution; it does not
test held-out context–key combinations or out-of-distribution generalization.

Report per-query accuracy and exact-table accuracy for every cell, equal-cell
macro accuracy, key-only and context-only majority oracles, and a conservative
single-field control that chooses the better oracle for each entire table. Estimate paired
95% confidence intervals using 2,000 bootstrap samples of whole tables, preserving
within-table query dependence. Macro intervals resample independently within each
cell and average the cell accuracies equally.

For a confirmed positive result, both backbones must have positive SMat-minus-native
macro and largest-cell differences whose paired confidence intervals exclude zero.
The largest-cell SMat-minus-single-field-control difference must also have a
positive paired 95% interval. This criterion is recorded before fresh evaluation.
Report ceiling
effects or losses in other cells rather than omitting them. These intervals concern
evaluation sampling conditional on training seed 123, not training-seed robustness.

If confirmation fails, preserve the complete outcome. Any later model selection
requires a newly locked selection and a new fresh split code; the failed confirmation
set becomes development evidence and cannot be reused as an independent test.

`confirm_joint_recall.py` performs the locked evaluation and saves raw predictions.
`collect_joint_recall_confirmation.py` renders the completed results. Completion
requires auditing the saved predictions, budgets, selected settings, and the
stated positive-comparison criteria; report files alone are not proof of success.
