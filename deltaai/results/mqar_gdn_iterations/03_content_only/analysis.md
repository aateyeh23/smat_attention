Rejected at epoch4: 56.6434375% vs GDN 60.65875%.
Removing the two fixed anchor writes leaves aggregate accuracy essentially
unchanged from trial02 (56.66375%). The measured positional routing bias was
real, but anchor removal alone does not explain or close the baseline gap.
Next test: score the same four reads from the writer's continuous hash address,
sharing the address parameters across sequence lengths. This directly trains
writer/query alignment and transfers routing updates between lengths.
Keep all writes content-based and all other trial03 choices unchanged.
