---
layout: page
title: "Experiments"
---

This page is a **visual overview** of the experiment suite. For exact definitions, loss functions, and equations, see the manuscript source in `paper/main.tex`.

## Quick navigation

- [EX1 — MultiGate vs SingleGate convergence](#ex1--multigate-vs-singlegate-convergence)
- [EX2 — Lissajous reconstruction (MultiGate vs SingleGate)](#ex2--lissajous-reconstruction-multigate-vs-singlegate)
- [EX3 — K-dependency](#ex3--k-dependency)
- [EX4 — Open vs Closed regime + crossover](#ex4--open-vs-closed-regime--crossover)
- [EX5 — Binding and stability](#ex5--binding-and-stability)
- [EX6 — Category detection after freezing](#ex6--category-detection-after-freezing)
- [EX7 — Relational time](#ex7--relational-time)
- [EX8 — Semantic Grounding via Convex Hull](#ex8--semantic-grounding-via-convex-hull)
- [EX9 — Syntax Inference via Barycentric Interpolation](#ex9--syntax-inference-via-barycentric-interpolation)

---

## EX1 — MultiGate vs SingleGate convergence

**Question.** Does a MultiGate router converge differently than a SingleGate baseline under matched settings?

**Figures.** Seed-mean convergence plots + aggregate summary.

![EX1 aggregate](assets/ex1/ex1_aggregate.png)

![Convergence avg MultiGate](assets/ex1/convergence_avg_MultiGate.png)

![Convergence avg SingleGate](assets/ex1/convergence_avg_SingleGate.png)

---

## EX2 — Lissajous reconstruction (MultiGate vs SingleGate)

**Question.** Under identical waveform families, does MultiGate improve reconstruction or routing stability versus SingleGate?

![EX2 MultiGate aggregate](assets/ex2/ex2_MultiGate_aggregate.png)

![EX2 MultiGate example](assets/ex2/ex2_MultiGate_lissajous.png)

![EX2 SingleGate aggregate](assets/ex2/ex2_SingleGate_aggregate.png)

![EX2 SingleGate example](assets/ex2/ex2_SingleGate_lissajous.png)

---

## EX3 — K-dependency

**Question.** How does discreteness / code usage behave as a function of the K setting?

![EX3 aggregate](assets/ex3/ex3_aggregate.png)

![K-dependency analysis](assets/ex3/k_dependency_analysis.png)

---

## EX4 — Open vs Closed regime + crossover

**Question.** How do errors evolve across regimes, and where does the crossover occur?

![EX4 aggregate](assets/ex4/ex4_aggregate.png)

![EX4 crossover](assets/ex4/ex4_crossover.png)

![EX4 noise floor](assets/ex4/ex4_noise_floor.png)

---

## EX5 — Binding and stability

**Question.** How does binding quality relate to stability duration and late-phase alignment?

![EX5 aggregate](assets/ex5/ex5_aggregate.png)

![EX5 binding](assets/ex5/ex5_binding.png)

---

## EX6 — Category detection after freezing

**Question.** After freezing the codebook, does MultiGate treat a withheld shape as categorically distinct?

![EX6 aggregate](assets/ex6/ex6_aggregate.png)

![EX6 reconstruction example](assets/ex6/ex6_reconstruction.png)

---

## EX7 — Relational time

**Question.** Do internal relational-time metrics track environment changes across worlds?

![EX7 aggregate](assets/ex7/ex7_aggregate.png)

![EX7 relational time](assets/ex7/ex7_relational_time.png)

---

## EX8 — Semantic Grounding via Convex Hull

**Question.** Can a withheld primitive be inferred after codebook freezing? Does success depend on convex hull geometry?

**Key Results (Table-based, no figures):**

| Condition | Withheld Similarity | Success Rate |
|-----------|---------------------|--------------|
| IN_HULL   | 0.979 ± 0.025       | 96.7% (29/30) |
| RANDOM    | 0.682 ± 0.370       | 53.3% (16/30) |
| OUT_HULL  | 0.701 ± 0.242       | 16.7% (5/30)  |

Statistical significance: Kruskal–Wallis H = 42.52, p = 5.85 × 10⁻¹⁰

---

## EX9 — Syntax Inference via Barycentric Interpolation

**Question.** Can a withheld composite (B+C) be reconstructed using only trained primitives and composites?

**Key Results (Table-based, no figures):**

| Condition | Withheld Similarity | Success Rate |
|-----------|---------------------|--------------|
| IN_HULL   | 0.890 ± 0.138       | 66.7% (20/30) |
| RANDOM    | 0.767 ± 0.253       | 33.3% (10/30) |
| OUT_HULL  | 0.742 ± 0.168       | 13.3% (4/30)  |

Statistical significance: Kruskal–Wallis H = 19.13, p = 7.00 × 10⁻⁵

---

For full details, see the manuscript: [10.5281/zenodo.18408862](https://doi.org/10.5281/zenodo.18408862)
