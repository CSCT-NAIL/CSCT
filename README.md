# CSCT: Clock-Selected Compression Theory
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18382368.svg)](https://doi.org/10.5281/zenodo.18382368)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official implementation of the paper:**

> **Intelligence within Bounds: Why Cognition Requires a Closed Convex Hull*
>
> *How do discrete symbolic representations emerge from continuous neural dynamics?*

---

## Overview

CSCT (Clock-Selected Compression Theory) is an axiomatic framework that explains how discrete symbols emerge from continuous signals through geometric constraints and phase-locked temporal selection.

### The 5 Axioms

| Axiom | Name | Description |
|-------|------|-------------|
| **A1** | Streams | Cognition operates on continuous, time-indexed streams |
| **A2** | Constructive Compression | Representations are constrained to convex combinations within a simplex |
| **A3** | Multi-Clock Factorization | Discrete events emerge via phase-locked clock selection |
| **A4** | Irreversible Anchor | External anchors impose thermodynamic directionality |
| **A5** | Barycentric Syntax | Complex operations arise from geometric interpolation, not symbol concatenation |

### Key Findings

- **Discretization emerges reliably** from continuous dynamics (EX1-3)
- **Irreversible anchors** outperform self-referential systems in long-term stability (EX4)
- **Feature binding** arises from shared phase without explicit concatenation (EX5)
- **Semantic grounding** requires convex-hull membership: 96.7% vs 16.7% success (EX8)
- **Syntactic composition** emerges as barycentric interpolation (EX9)
- **Ungrounded Symbol Acquisition**: discrete codes can be assigned without reconstructable meaning

---

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/CSCT-NAIL/csct.git
cd csct

# Install dependencies
pip install -r requirements.txt

# Run a single experiment
python csct_ex1_waveforms.py --seed 42

# Run all experiments
python csct_suite.py --run all --seeds 30
```

### Dependencies

```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0  # optional, for summary tables
```

---

## Experiments

### Overview

| Experiment | Axiom | Description | Key Metric |
|------------|-------|-------------|------------|
| **EX1** | A1, A3 | Single-channel discretization | MSE, Unique codes |
| **EX2** | A3 | Multi-channel relational encoding | Lissajous reconstruction |
| **EX3** | A2 | Codebook size (K) dependency | MSE vs K curve |
| **EX4** | A4 | Irreversible anchor stability | Long-term drift |
| **EX5** | A3 | Feature binding via shared clock | PLV (Phase-Locking Value) |
| **EX6** | A2 | Category recognition (frozen codebook) | MSE ratio, JSD |
| **EX7** | A3 | Relational internal time | Event rate dilation |
| **EX8** | A2, A5 | Semantic grounding (convex hull) | IN_HULL vs OUT_HULL |
| **EX9** | A5 | Syntax inference (barycentric) | Composition success rate |

### Running Individual Experiments

```bash
# EX1: Single-channel waveform discretization
python csct_ex1_waveforms.py --wave sine --gate single --seed 42

# EX2: Multi-channel relational (Lissajous)
python csct_ex2_relational.py --seed 42

# EX3: Codebook size dependency
python csct_ex3_kdependency.py --k-values 2,4,8,16 --seed 42

# EX4: Anchor stability (noise vs drift)
python csct_ex4_noise_floor.py --noise-level 0.1 --seed 42

# EX5: Feature binding
python csct_ex5_binding.py --seed 42

# EX6: Category recognition
python csct_ex6_category_recognition.py --seed 42

# EX7: Relational time
python csct_ex7_relational_time.py --seed 42

# EX8: Semantic grounding (meaning emergence)
python csct_ex8_meaning.py --condition IN_HULL --seed 42

# EX9: Syntax inference
python csct_ex9_syntax.py --seed 42
```

### Running the Full Suite

```bash
# Run all experiments with 30 seeds
python csct_suite.py --run all --seeds 30

# Run specific experiment
python csct_suite.py --run ex8 --seeds 30

# Run with custom parameters
python csct_suite.py --run all --seeds 10 --n-clocks 8 --device cuda
```

### Output Structure

```
results/
├── ex1/
│   ├── sine/single/seed0/
│   │   ├── ex1_metrics.csv
│   │   ├── ex1_reconstruction.png
│   │   └── ex1_convergence.png
│   └── ...
├── ex8/
│   ├── IN_HULL/seed0/
│   │   ├── ex8_metrics.csv
│   │   └── ex8_extraction.png
│   ├── RANDOM/seed0/
│   └── OUT_HULL/seed0/
└── summary/
    ├── ex1_summary.csv
    ├── ex8_summary.csv
    └── aggregate_figures/
```

## Reproducing Paper Results

### Main Results (Table 13 in paper)

```bash
# EX8: Semantic Grounding
python csct_suite.py --run ex8 --seeds 30

# Expected results:
#   IN_HULL:  96.7% success (29/30)
#   RANDOM:   53.3% success (16/30)
#   OUT_HULL: 16.7% success (5/30)
```

### Key Figures

```bash
# Generate aggregate figures
python csct_suite.py --run all --seeds 30

# Figures are saved to results/summary/aggregate_figures/
```

---

## Citation
@article{csct2026,
  title={Intelligence within Bounds: Why Cognition Requires a Closed Convex Hull},
  author={Higuchi, Naoki},
  year={2026},
  journal={Preprint},
  note={Work in progress},
  url={https://github.com/CSCT-NAIL/CSCT}
}

**Software Citation:**

> CSCT-NAIL. (2026). CSCT-NAIL/CSCT: CSCT Engine: Initial Release (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.18382368
---

## Project Structure

```
csct/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── csct_engine.py                 # Core CSCT Engine implementation
├── csct_suite.py                  # Unified experiment runner
├── csct_ex1_waveforms.py          # EX1: Single-channel discretization
├── csct_ex2_relational.py         # EX2: Multi-channel relational
├── csct_ex3_kdependency.py        # EX3: K-dependency
├── csct_ex4_noise_floor.py        # EX4: Anchor stability
├── csct_ex5_binding.py            # EX5: Feature binding
├── csct_ex6_category_recognition.py # EX6: Category recognition
├── csct_ex7_relational_time.py    # EX7: Relational time
├── csct_ex8_meaning.py            # EX8: Semantic grounding
├── csct_ex9_syntax.py             # EX9: Syntax inference
└── docs/                          # Documentation (GitHub Pages)
    ├── index.html
    └── ...
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact the author.
