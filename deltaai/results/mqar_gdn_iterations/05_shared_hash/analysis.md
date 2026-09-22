Rejected at epoch4:43.37625% vs60.65875%.
Hash sharing alone also regresses from trial03's56.64%; therefore the trial04
regression cannot be attributed solely to replacing the categorical reader.
Do not retain cross-length sharing. The writer context convolution currently
receives only surrogate routing/occupancy gradients; transport uses a separate
GDN Q/K basis. Next connect that learned causal context to the actual transported
content keys through a tied memory Q/K projection, a task-general way to learn
which information should label a write. No key/value offsets or labels are used.
