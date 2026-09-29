# HARP -- Reproducibility Artifact

This document is the artifact description required by the **IEEE TPDS Reproducibility Evaluation Process**. It accompanies the code in this repository and follows the structure required by the [TPDS Reproducibility Author Instructions](https://www.computer.org/csdl/journal/td/misc/299908?title=Reproducibility%20Author%20Instructions&periodical=IEEE%20Transactions%20on%20Parallel%20and%20Distributed%20Systems).

For a general introduction to HARP (installation, CLI usage, output format), see [README.md](README.md). This document focuses specifically on what a reviewer needs to reproduce the experiments reported in the paper.

---

## 1. Artifact Identification

**Title:** HARP: Heterogeneous Analog-Digital Resource-Aware Performance and Scheduling Framework for Transformer Acceleration

**Authors:** Elena Ferro (elf@zurich.ibm.com), Hadjer Benmeziane (Hadjer.Benmeziane@ibm.com), and Irem Boybat (ibo@zurich.ibm.com) -- IBM Research Europe, 8803 Rueschlikon, Switzerland.

**Venue:** IEEE Transactions on Parallel and Distributed Systems (TPDS), Vol. 37, Issue 9, September 2026. [https://ieeexplore.ieee.org/document/11576587](https://ieeexplore.ieee.org/document/11576587)

**Abstract:**
As deep learning models grow in size and complexity, data movement, rather than raw compute, has become the dominant performance bottleneck. Analog Compute-In-Memory (ACIM) mitigates this by performing multiply-accumulate operations directly in memory, eliminating repeated weight transfers and achieving order-of-magnitude energy efficiency gains. However, their high weight write cost and lack of native support for non-linear operations require tight integration with digital units that support diverse execution modes such as vector-style and data-parallel processing. Despite growing interest, there is still a lack of optimized tools to evaluate such heterogeneous ACIM-based systems at full-network scale. To address this, we present HARP, a unified framework for mapping and scheduling transformer workloads on heterogeneous systems combining ACIM units with digital accelerators supporting multiple execution flows. HARP explores latency-, area-, and balanced mapping strategies for ACIM, exploiting crossbar weight reuse to preserve efficiency, and constructs hybrid schedules that exploit both inter- and intra-layer parallelism through fine-grained pipelining, overlapping execution across analog and digital units while respecting each unit's dataflow constraints. By accepting arbitrary architectural parameters, HARP serves as both a fast and accurate performance estimator and a tool for rapid design-space exploration or hardware-aware neural architecture search. Evaluated on transformers, HARP reduces end-to-end latency by up to 14.54*x* over a non-pipelined baseline and 7.59*x* over a single-vector pipeline adopted in state-of-the-art ACIM simulators.

**Role of the artifact:** The artifact *is* HARP itself -- the mapping, scheduling and simulation framework described in the paper. It contains everything needed to (i) simulate the heterogeneous ACIM+digital hardware target across mapping/pipelining/parallelism configurations, and (ii) regenerate the quantitative results and figures (Figs. 6-13) reported in the evaluation section, directly from the paper's TABLE III architecture parameters. There is no separate hardware prototype: HARP's own predictive cost models (RTL-calibrated random-forest predictors for the digital units, analytical models for ACIM) are the mechanism by which the paper's latency/throughput/energy numbers are produced, so running this code *is* reproducing the experiments.

---

## 2. Artifact Dependencies and Requirements

**Hardware:** No special or proprietary hardware is required. HARP is a pure-Python performance model/scheduler -- it does not execute on the analog or digital accelerators it simulates, and it does not require a GPU. Any modern x86_64 or ARM64 machine with a few free CPU cores is sufficient.

**Operating systems required:** GNU/Linux (also verified on macOS). `main.py` selects a headless-safe Matplotlib backend (`Agg`) unless `--display_plots true` is passed, so it also runs unmodified over SSH, in CI, or in a container.

**Software:**
- Python 3.10+
- Packages pinned in [requirements.txt](requirements.txt): `matplotlib`, `networkx`, `joblib`, `pandas`, `colorama`, `scikit-learn==1.6.1` (pinned because the PMCA latency predictors in `pmca_prediction_model/*.pkl` were serialized with that exact version)

**Input data:** None external. Transformer architectures are generated in code (`models/`) from the parameters in the paper's TABLE III; the RTL-simulated PMCA latency predictors ship with the repository as pickled models in `pmca_prediction_model/`.

---

## 3. Artifact Installation and Deployment Process

**Install (~2-5 minutes, no compilation step):**
```bash
git clone https://github.com/IBM/HARP.git
cd HARP
python3 -m venv harp
source harp/bin/activate
pip install -r requirements.txt
```

**Deploy:** There is no separate deployment step -- `main.py` and the scripts under `regression_checks/` run directly from the checkout.

**Sanity check (~105 seconds):** Before attempting full figure reproduction, verify the installation with the fast regression check, restricted to one model suite (MobileBERT, 6 pipelining/parallelism configs), which asserts latency against reference values:
```bash
python3 regression_checks/check_models_regression.py --models MobileBERT
```
A clean run reports `6/6 (100.0%)`. The full check covers all 7 model suites x 6 configs = 42 runs and takes ~45 minutes:
```bash
python3 regression_checks/check_models_regression.py
```

> **Concurrency warning:** both `check_models_regression.py` and `reproduce_figure.py` temporarily rewrite `nodes/accelerator_config.py` as shared global state and restore it on exit. Never run two invocations against the *same checkout* concurrently -- they will race. To parallelize across machines, use a separate clone of the repository per concurrent job.

---

## 4. Reproducibility of Experiments

**Workflow.** `regression_checks/reproduce_figure.py` reproduces the paper's evaluation figures one at a time: it applies the correct TABLE III architecture, runs the fixed flag set from the paper with the figure's swept parameter, prints a metrics table, and saves a PNG plot laid out like the paper figure.

```bash
python3 regression_checks/reproduce_figure.py --list          # show all available figures
python3 regression_checks/reproduce_figure.py --figure 6      # fastest figure (~4s) -- good first smoke test
python3 regression_checks/reproduce_figure.py --figure 7      # run + plot one figure
python3 regression_checks/reproduce_figure.py --figure 8 --no-plot   # table only, skip the PNG
python3 regression_checks/reproduce_figure.py --figure 10 --replot   # redraw from cached results, no re-simulation
```

**Command for every figure**, using the same `--figure <key>` pattern (see the mapping table below for what each key reproduces and how long it takes from scratch):

```bash
python3 regression_checks/reproduce_figure.py --figure 6
python3 regression_checks/reproduce_figure.py --figure 7
python3 regression_checks/reproduce_figure.py --figure 8
python3 regression_checks/reproduce_figure.py --figure 9
python3 regression_checks/reproduce_figure.py --figure 10
python3 regression_checks/reproduce_figure.py --figure 11
python3 regression_checks/reproduce_figure.py --figure 12     # runs 12a + 12b automatically if needed
python3 regression_checks/reproduce_figure.py --figure 13

# Reproduce every figure sequentially in one checkout (~84h from scratch, see below;
# do not parallelize these within the same checkout -- see the concurrency warning above):
for fig in 6 7 8 9 10 11 12 13; do
    python3 regression_checks/reproduce_figure.py --figure "$fig"
done
```

**Figure &rarr; paper mapping and expected runtime** (measured wall-clock time for a full, uncached run on the authors' reference machine; `reproduce_figure.py` writes its own `regression_checks/figures/run_timing.json` locally each time you run a figure from scratch, but that file is machine-dependent and gitignored, so it is not included in the repository):

| Key | Paper figure | What it shows | Simulation runs | Measured runtime |
|---|---|---|---|---|
| `6` | Fig. 6 | Analog mappings for a small illustrative test model (2 blocks x 6 FC layers): (a) latency, (b) area, (c) area/latency balance opt_level | 3 | ~4 s |
| `7` | Fig. 7 | MobileBERT performance across analog mapping optimizations (SL=128, 384) | 12 | ~13 min |
| `8` | Fig. 8 | Pipeline/parallelism optimizations across 5 models (SL=128, 384) | 60 | ~2.5 h |
| `9` | Fig. 9 | Throughput/energy efficiency vs. link bandwidth, BERT-B/BERT-L | 16 | ~1.7 h |
| `10` | Fig. 10 | Chunk-size optimization across 5 models (SL=128, 384) | 30 | ~71 h |
| `11` | Fig. 11 | Chunk optimization across pipelining techniques, MobileBERT | 12 | ~8.4 h |
| `12` | Fig. 12 | Single-token generation: (a) pipeline/parallelism techniques for GPT-2 Small & T5 Small, (b) on-chip SRAM capacity sweep for GPT-2 Small | 23 | ~7.4 min |
| `13` | Fig. 13 | Throughput/energy efficiency vs. number of generated tokens, autoregressive decoding under XB-Pipe | 36 | ~16.3 min |

Total sequential wall-clock for a from-scratch reproduction of every figure is **~84 hours**, dominated by Fig. 10's chunk-size search. Figures are independent of one another (each applies its own TABLE III architecture).

> **Fig. 5 is out of scope for this artifact.** Fig. 5 compares several *low-level latency predictor* techniques for the digital units (e.g. linear vs. random forest) against each other. HARP itself, and this repository, ship and use only the one predictor chosen for the rest of the paper's evaluation -- an RTL-calibrated random-forest model (`pmca_prediction_model/`) -- not the other predictors compared in that figure. Reproducing Fig. 5 is therefore outside the scope of this artifact.

> **Fig. 12 is generated from Figs. 12(a) and 12(b).** `--figure 12` combines the pipeline/parallelism panel (internally `12a`) and the SRAM sweep panel (internally `12b`) into the final Fig. 12 PNG. It auto-runs whichever of the two hasn't been simulated yet (cache missing), so a single `--figure 12` works from scratch; `--figure 12a`/`--figure 12b` can also be run individually to inspect or re-run just one panel. 

> **Fig. 14 is out of scope for this artifact.** Figs. 6-13 are produced directly by HARP itself, and this artifact reproduces all of them. Fig. 14, by contrast, reports a Neural Architecture Search / Design Space Exploration (NAS/DSE) case study in which HARP is invoked as the per-candidate performance/energy estimator inside an *outer* search loop, rather than run standalone -- HARP is a component used *by* that search tool, not the subject of the figure itself. Reproducing Fig. 14 would require standing up that external NAS/DSE search infrastructure, which is not part of this repository and outside HARP's own scope. 


**Recommended reviewer path, given the above:**
1. **Quick verification (< 20 min):** run figures `6` (~4s, analog mapping only), `12`, and `7` from scratch. Together these exercise the full pipeline (analog mapping, scheduling, RF latency prediction, plotting) at low cost.
2. **Cache-assisted check (seconds):** this repository ships the raw metrics already computed for every figure in `regression_checks/figures/fig<N>_results.json`. Running any `--figure <N> --replot` regenerates that figure's PNG from those cached numbers in seconds, letting a reviewer confirm the plotting/formatting pipeline matches the paper without re-simulating.
3. **Full from-scratch reproduction:** run `--figure <N>` without `--replot` (the default) -- this always re-simulates every run for that figure regardless of whether a cache already exists, and overwrites `fig<N>_results.json` with the fresh results. Budget with the runtimes above; distribute the long figures (`10`, `11`) across separate nodes/checkouts if reviewing under time constraints.

**Expected results.** Each run prints a metrics table to stdout and saves `regression_checks/figures/fig<N>_reproduced.png`, laid out to match the corresponding paper figure panel-for-panel (same models/configurations on the x-axis, same throughput/energy-efficiency metrics on the y-axis). Comparing `fig<N>_reproduced.png` against Fig. `<N>` in the published paper is the direct reproducibility check; the underlying numbers backing each plot are in the paired `fig<N>_results.json`.

---

## 5. Citation

If you use HARP, please cite:

```bibtex
@ARTICLE{HARP_TPDS2026,
  author={Ferro, Elena and Benmeziane, Hadjer and Boybat, Irem},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  title={HARP: Heterogeneous Analog-Digital Resource-Aware Performance and Scheduling Framework for Transformer Acceleration},
  year={2026},
  volume={37},
  number={9},
  pages={2107-2121},
  doi={10.1109/TPDS.2026.3706975}}
```

---

## 6. Other notes

- License: Apache-2.0 (see [LICENSE](LICENSE)).
- DOI: [10.5281/zenodo.23018118](https://doi.org/10.5281/zenodo.23018118) (Zenodo deposit archiving the `v1.0.0` GitHub release).
- Questions about the artifact can be directed to elf@zurich.ibm.com (primary contact), or ibo@zurich.ibm.com (Irem Boybat, secondary contact).
