---
layout: page
title: "Reproduction checklist"
---

> Code repository: [https://github.com/CSCT-NAIL/csct](https://github.com/CSCT-NAIL/csct)

## Minimal steps

1. **Clone** the code repository
   ```bash
   git clone https://github.com/CSCT-NAIL/csct.git
   cd csct
   ```

2. **Run the suite**
   - Use the provided suite driver (e.g., `csct_suite.py`) to generate per-seed CSVs and figures.

3. **Generate LaTeX tables**
   - Each experiment has a `make_ex*_table.py` script that reads `summary.csv` and writes `ex*_table.tex`.

4. **Build the paper**
   - `cd paper && latexmk -pdf main.tex`

## Expected outputs

- `results_suite/ex*/summary.csv` (per-seed rows)
- `paper/results_ex*/...png` (figures)
- `paper/sections/...` (manuscript sections)

## Notes

- The scripts are designed so that **numbers used in the text** are printed by CLI summaries (EX1-style), while **tables** are generated separately from CSV.
