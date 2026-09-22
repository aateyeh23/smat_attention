Rejected at epoch4:37.00125% vs60.65875%. Address-derived
read scoring also regresses without sharing hashes across lengths.
Return to trial07's categorical reader. The current write surrogate
always interpolates floor(s) and floor(s)+1, although floor(s) is the
hard bucket for the entire unit interval. Next test a symmetric local
surrogate centered on bucket centers. This preserves hard forwards,
while changing only the write-hash task gradient; marginal occupancy
regularization and payload gradients remain unchanged.
