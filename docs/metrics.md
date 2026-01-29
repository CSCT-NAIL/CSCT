---
layout: page
title: "Metrics & reporting"
---

## Reporting vs inference

CSCT distinguishes:

- **Operational reporting criterion** (binary pass/fail): e.g., `withheld_sim ≥ 0.90`.
- **Statistical analysis** on the **continuous** metric (e.g., `withheld_sim`) to avoid threshold artifacts.

This separation is used consistently across EX6–EX9.

## Common metrics

- **withheld_sim**: similarity score for withheld target (primitive or composite).
- **mean_trained_sim**: sanity check that trained patterns reconstruct well.
- **reconstruction MSE**: distortion of reconstructed signal.
- **transition rate**: how often the discrete routing state changes.
- **maxP / entropy**: confidence and dispersion of gate/code selections.
- **cluster metrics** (e.g., ARI/NMI): separability of trained vs withheld from internal features.

## EX8/EX9 specific

- **Hull modes** (`IN_HULL`, `OUT_HULL`, `RANDOM`) define the geometry of withheld targets relative to trained primitives.
- Nonparametric tests (Kruskal–Wallis; post-hoc Mann–Whitney) are used when distributions are non-Gaussian or heavy-tailed.
