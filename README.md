# Supersonic Cargo Transport — Environmental and Cost Analysis

Reproducible analysis code for the manuscript:

> **Environmental and Cost Analysis of Supersonic Cargo Transport: A Comparative Study with Subsonic Aircraft and Other Freight Modes**
> Amir Avitan and Chen Giladi, Department of Mechanical Engineering, Sami Shamoon College of Engineering (SCE), Ashdod, Israel.
> Submitted to *Energies* (MDPI).

This repository contains the Python scripts that regenerate **every quantitative result, figure, and table** in the paper: a screening-level techno-environmental and cost comparison of a first-generation supersonic freighter against subsonic air freight, trucking, rail, and maritime shipping, normalized to a one-ton functional unit.

## Method summary

- **Fuel-based emissions model** — block fuel from a Breguet supersonic performance grid converted to CO2/H2O/NOx via emission factors and literature emission indices.
- **Direct operating cost (DOC) decomposition** — fuel, maintenance, capital, crew, and fees, allocated to a one-ton functional unit.
- **Monte Carlo uncertainty propagation** — load factor, fuel price, and emission-factor variability (`numpy` seeded for determinism).

## Repository contents

| Script (`scripts/`) | Reproduces |
|---|---|
| `emissions_data_graph_code.py` | Figure 1 — cross-modal emissions comparison |
| `monte_carlo_summary.py` | Figure 2 — Monte Carlo uncertainty summary; the MC CO2/DOC percentile intervals |
| `parameter_heatmap.py` | Figure 3 — CO2-intensity parameter heatmap |
| `vot_breakeven.py` | Figure 4 — value-of-time break-even curves |
| `parametric_summary.py` | `parametric_summary.csv` — single source of truth for the parametric headline numbers (scenarios, envelope, stage lengths) |
| `parametric_summary.csv` | Reference output of `parametric_summary.py` (committed for convenience) |

## Requirements

- Python 3.10+ (developed on 3.12)
- See `requirements.txt` (`numpy`, `matplotlib`)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce

```bash
cd scripts
python parametric_summary.py        # writes parametric_summary.csv
python emissions_data_graph_code.py # Figure 1
python monte_carlo_summary.py       # Figure 2 (deterministic; numpy seed fixed)
python parameter_heatmap.py         # Figure 3
python vot_breakeven.py             # Figure 4
```

The Monte Carlo routine fixes the `numpy` random seed, so re-running reproduces the reported percentile intervals exactly.

## License

Code is released under the MIT License (see `LICENSE`). Please cite the manuscript if you use this code (see `CITATION.cff`).
