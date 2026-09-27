#!/usr/bin/env python3
#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""
reproduce_figure.py

Reproduce individual HARP paper figures/tables by name: applies the right
TABLE III architecture, runs the fixed flag set with the figure's swept
parameter, prints a metrics table, and (by default) saves a PNG plot laid
out like the paper figure.

Figures are added incrementally as each one is verified against the paper
numbers -- see FIGURES below for what's currently supported.

Usage
-----
$ python regression_checks/reproduce_figure.py --list
$ python regression_checks/reproduce_figure.py --figure 7
$ python regression_checks/reproduce_figure.py --figure 7 --no-plot
$ python regression_checks/reproduce_figure.py --figure 8 --replot   # re-draw from the
                                                                      # cached results, no re-simulation

WARNING: this rewrites nodes/accelerator_config.py as it goes, like
check_models_regression.py. The original contents are restored on exit. Never
run two invocations of this script (or check_models_regression.py) at once --
they share that one config file as global state and will race.

Every figure caches its raw metrics to regression_checks/figures/fig<key>_results.json
after a run, so styling-only tweaks can be redrawn with --replot instead of
re-running every simulation.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_models_regression import (
    apply_model_arch, parse_metrics, PYTHON, REPO_ROOT, CONFIG_PATH,
    MODEL_SUITES, PIPE_CONFIGS, run_config,
)
from set_model_arch import set_value as _sma_set_value

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
STATUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reproduce_status.txt")


def _write_status(line):
    try:
        with open(STATUS_FILE, "w") as f:
            f.write(line + "\n")
    except OSError:
        pass


EXTRA_PATTERNS = {
    "acim_used":     r"Total ACIM tiles used:\s+(\d+)\s*/",
    "pmca_red_used": r"Total PMCA_RED used:\s+(\d+)\s*/",
    "pmca_used":     r"Total PMCA used:\s+(\d+)\s*/",
}

# Per-model, per-config bar colors, in PIPE_CONFIGS order
# (Serial, IL-Par, IL-Pipe, XL-Pipe, IL+XL-Pipe, XB-pipe). Shared across figures
# so the same model+config always gets the same color.
MODEL_COLORS = {
    "MobileBERT": ["#EBDCE4", "#DDC1CE", "#B67D98", "#8D506E", "#713F57", "#4B2A39"],
    "BERT_B":     ["#F7D5D1", "#EDB6B2", "#DA6659", "#CF3F2F", "#BC392A", "#912C20"],
    "BERT_L":     ["#D1EDFB", "#ABDDF3", "#4FB5E4", "#1E8FC2", "#18719A", "#0F445C"],
    "GPT2_small": ["#EDEDED", "#D3D3D3", "#A8A8A8", "#818181", "#636363", "#4F4F4F"],
    "T5_encoder": ["#D2E4E6", "#A4CBCB", "#77B2B1", "#519798", "#2A6465", "#1E4B4B"],
}

# key -> figure spec.
#   arch_model : TABLE III row to apply via set_model_arch.py
#   sls        : sequence lengths to run (one panel per SL)
#   fixed_args : CLI args held constant across the sweep
#   sweep_flag : the CLI flag varied across the x-axis
#   sweep      : [(bar label, flag value), ...]
FIGURES = {
    "6": {
        "title": "Fig. 6. Analog mappings: (a) Latency optimized; (b) area optimized; "
                 "(c) area/latency balance.",
        "kind": "analog_mapping",
        "model": "dumpModel",
        "arch_model": "MobileBERT",
        "opt_levels": [("a", "latency"), ("b", "area"), ("c", "area_latency_balance")],
    },
    "7": {
        "title": "Fig. 7 -- MobileBERT performance for different analog mapping optimizations "
                 "for (a) SL=128, (b) SL=384, (c) for both SLs.",
        "arch_model": "MobileBERT",
        "sls": [128, 384],
        "fixed_args": ["--model", "MobileBERT", "--autoregressive", "false",
                       "--parall_type", "intra_layer", "--pipe_type", "inter_block",
                       "--chunk_size_d", "8"],
        "sweep_flag": "--opt_level",
        "sweep": [("Area Opt", "area"), ("Lat Opt", "latency"),
                  ("Area/Lat", "area_latency_balance")],
    },
    "8": {
        "title": "Fig. 8 -- Pipeline and parallelism optimizations for different models: "
                 "(a) SL=128, (b) SL=384. The green and red arrows highlight the relative "
                 "performance of XB-Pipe compared to the serial implementation.",
        "kind": "model_sweep",
        # Suite names from MODEL_SUITES (check_models_regression.py) -- GPT2_small and
        # T5_encoder here mean the non-autoregressive suites (autoregressive=false).
        "models": ["MobileBERT", "BERT_B", "BERT_L", "GPT2_small", "T5_encoder"],
        "model_labels": {
            "MobileBERT": "MobileBERT",
            "BERT_B": "BERT-B",
            "BERT_L": "BERT-L",
            "GPT2_small": "GPT-2 Small",
            "T5_encoder": "T5 Small",
        },
        "model_colors": MODEL_COLORS,
        "sls": [128, 384],
        # (sl, metric_key) -> explicit (ymin, ymax), overriding the auto-computed range.
        "ylim_overrides": {
            (128, "energy_eff"): (1, 410),
        },
        "hide_top_right_spines": True,
        "combined_caption": "Fig. 8.   Pipeline and parallelism optimizations for different "
                            "models: (a) SL=128, (b) SL=384. The green and red arrows highlight "
                            "the relative performance of XB-Pipe compared to the serial "
                            "implementation.",
        "model_label_dy": -0.22,
    },
    "9": {
        "title": "Fig. 9 -- Throughput (a) and energy efficiency (b) of Serial and XB-Pipe "
                 "execution for BERT-B and BERT-L under varying bandwidth (BW).",
        "kind": "bw_sweep",
        "models": ["BERT_B", "BERT_L"],
        "model_labels": {"BERT_B": "BERT-B", "BERT_L": "BERT-L"},
        "configs": ["Serial", "XB-pipe"],
        "bandwidths": [512, 384, 256, 128],
        "sl": 384,
        # Reuse the same per-model colors as Fig. 8: index 0 = Serial, index 5 = XB-pipe.
        "model_colors": {
            model: {"Serial": MODEL_COLORS[model][0], "XB-pipe": MODEL_COLORS[model][5]}
            for model in ["BERT_B", "BERT_L"]
        },
    },
    "10": {
        "title": "Fig. 10 -- Chunk-size optimization across models at SL=128 (a) and SL=384 "
                 "(b) using XB-Pipe and IL-Par. Best chunk is highlighted in yellow.",
        "kind": "chunk_opt_sweep",
        "models": ["MobileBERT", "BERT_B", "BERT_L", "GPT2_small", "T5_encoder"],
        "model_labels": {
            "MobileBERT": "MobileBERT",
            "BERT_B": "BERT-B",
            "BERT_L": "BERT-L",
            "GPT2_small": "GPT-2 Small",
            "T5_encoder": "T5 Small",
        },
        "sls": [128, 384],
        # chunk_opt's own search never tries these -- run them separately.
        "extra_chunks": [1, 4],
        # Cap on which chunk_opt-discovered chunk sizes to keep/display per SL
        # (chunk_opt's coarse sweep goes all the way to SL itself, e.g. 144/208/.../384
        # for SL=384, which the reference figure doesn't plot).
        "display_chunk_cap": {128: 128, 384: 80},
        # One solid color per model (reusing each model's XB-pipe shade from Fig. 8/9),
        # all chunks share it except the best chunk, highlighted in yellow.
        "model_colors": {model: MODEL_COLORS[model][5] for model in
                          ["MobileBERT", "BERT_B", "BERT_L", "GPT2_small", "T5_encoder"]},
        "best_color": "#F5D76E",
        # sl -> explicit y-axis max (overrides the auto-computed top*1.3).
        "ylim_overrides": {128: 60, 384: 480},
    },
    "11": {
        "title": "Fig. 11 -- Chunk optimization for different pipelining techniques for "
                 "MobileBERT: (a) SL=128 and (b) SL=384.",
        "kind": "chunk_opt_pipe_sweep",
        "model": "MobileBERT",
        # (group label, --pipe_type value). --parall_type is always intra_layer.
        "pipe_groups": [
            ("Inter-Layer", "inter_layer"),
            ("Inter- and Intra- Layer", "inter_intra_layer"),
            ("Inter-block", "inter_block"),
        ],
        "sls": [128, 384],
        "extra_chunks": [1, 4],
        "display_chunk_cap": {128: 128, 384: 80},
        # Reuse MobileBERT's existing palette (Fig. 8/9/10): lighter shades for the two
        # new pipe types, the same darkest shade Fig. 10 already used for Inter-block/XB-pipe.
        "group_colors": {
            "Inter-Layer": MODEL_COLORS["MobileBERT"][1],
            "Inter- and Intra- Layer": MODEL_COLORS["MobileBERT"][3],
            "Inter-block": MODEL_COLORS["MobileBERT"][5],
        },
        "best_color": "#F5D76E",
        # sl -> explicit y-axis max (overrides the auto-computed top*1.3).
        "ylim_overrides": {128: 45, 384: 300},
    },
    "12a": {
        "title": "Fig. 12(a) -- Single token generation: different pipeline and "
                 "parallelism techniques for GPT-2 Small and T5 Small.",
        "kind": "model_sweep",
        # Reuse the exact autoregressive single-token suites from check_models_regression.py.
        "models": ["GPT2_small_AR", "T5_decoder"],
        "model_labels": {"GPT2_small_AR": "GPT-2 Small", "T5_decoder": "T5 Small"},
        "model_colors": {"GPT2_small_AR": MODEL_COLORS["GPT2_small"],
                          "T5_decoder": MODEL_COLORS["T5_encoder"]},
        # No SL sweep here -- single-token generation is always SL=1 (each suite's own
        # common_args already fixes --sl 1 and --prefill_size); one panel, letter "a".
        "sls": [1],
        # Only 2 models here (vs. Fig. 8's 5) -- the default 18in width leaves the bars
        # sparse with a big gap between the two model groups, so use a narrower figure
        # and pull the bars themselves closer together too.
        "figsize": (7, 4.5),
        "bar_width": 0.58,
        "bar_step": 0.65,
        "model_gap": 0.8,
        # (sl, metric_key) -> explicit (ymin, ymax), overriding the auto-computed range.
        "ylim_overrides": {(1, "throughput"): (0, 710), (1, "energy_eff"): (0, 580)},
        "arrow_label_dx": 0.95,
        "arrow_label_y_mult": {"throughput": 0.99, "energy_eff": 1.06},
        "suptitle": "Single token generation: (a) different pipeline and\n"
                    "parallelism techniques for GPT-2 Small and T5.",
        # Shared outer label spanning all model groups, drawn below their own labels
        # (e.g. "Token 1" -- the generation step these single-token runs represent).
        "group_label": "Token 1",
        "model_label_dy": -0.27,
        "group_label_dy": -0.40,
    },
    "12b": {
        "title": "Fig. 12(b) -- Impact of on-chip SRAM capacity under XB-Pipe for GPT-2 Small.",
        "kind": "sram_sweep",
        # Same suite/config as Fig. 12(a)'s GPT-2 Small (single-token, autoregressive).
        "model": "GPT2_small_AR",
        "sram_values": list(range(1, 12)),  # 1..11 MiB (SRAM_TILE_CONFIG["size"] is 1 MiB/tile)
        "throughput_color": "#8B1A1A",
        "energy_eff_color": "#1B3B5F",
        "group_label": "Token 1",
    },
    "12": {
        "title": "Fig. 12 -- Single token generation: (a) different pipeline and "
                 "parallelism techniques for GPT-2 Small and T5, (b) impact of "
                 "on-chip SRAM capacity under XB-Pipe for GPT-2 Small.",
        # Assembles Fig. 12(a) + Fig. 12(b) into one image. Has no "run" path of its
        # own ("kind": "combine" is handled directly in main(), not via run_figure())
        # -- combine_fig12() runs whichever of 12a/12b's simulations are missing on
        # demand, so `--figure 12` alone works end to end from scratch.
        "kind": "combine",
    },
    "13": {
        "title": "Fig. 13 -- Throughput (a) and energy efficiency (b) for different "
                 "number of generated tokens during autoregressive decoding under XB-Pipe.",
        "kind": "token_sweep",
        # Same suites/config as Fig. 12(a), swept over --prefill_size instead of held fixed:
        # prefill = each suite's own base prefill + (n_tokens - 1) -- e.g. GPT2_small_AR's
        # base is 512 (--> 512, 513, 519, 639, ... for n_tokens 1, 2, 8, 128, ...);
        # T5_decoder's base is 0 (--> 0, 1, 7, 127, ... ), matching how its "prefill_size 0"
        # already represents an effective prefill of 1 (n_tokens=1) -- see main.py/README.
        "models": ["GPT2_small_AR", "T5_decoder"],
        "model_labels": {"GPT2_small_AR": "GPT-2 Small", "T5_decoder": "T5 Small"},
        "model_colors": {"GPT2_small_AR": MODEL_COLORS["GPT2_small"][3],
                          "T5_decoder": MODEL_COLORS["T5_encoder"][5]},
        "base_prefill": {"GPT2_small_AR": 512, "T5_decoder": 0},
        "n_tokens": [1, 2, 8, 128, 512, 1024, 2048, 3072, 4096],
        "group_label": "# generated tokens",
    },
}

CHUNK_MARKER_RE = re.compile(r"Testing chunk size:\s*(\d+)")
# Python's f"{x:.3f}" formats a non-finite float as bare "inf"/"-inf"/"nan" (no digits),
# so the value group must accept those too, not just [\d.]+.
LATENCY_MARKER_RE = re.compile(r"->\s*Latency:\s*(-?inf|nan|[\d.]+)\s*ms")


def _parse_chunk_opt_latencies(stdout):
    """Every (chunk_size, latency) pair `optimize_chunk_size[_v2]` tried, in the order
    tested (analog_scheduler.py's "Testing chunk size: N" / "-> Latency: X ms").
    Infeasible trials (chunk too large for the model's tile pool) report inf/nan
    latency and are dropped -- they were never real candidates.

    The two lines are NOT adjacent -- thousands of lines of per-layer scheduling
    output sit between them for a single chunk trial -- so they can't be matched
    with one regex; instead pair the two marker lists positionally. This is safe
    because the two prints strictly alternate 1:1 in the source (each trial prints
    its "Testing" line, then unconditionally its own "-> Latency" line before the
    next trial's "Testing" line ever appears).
    """
    chunks = [int(c) for c in CHUNK_MARKER_RE.findall(stdout)]
    latencies = [float(l) for l in LATENCY_MARKER_RE.findall(stdout)]
    if len(chunks) != len(latencies):
        raise ValueError(f"chunk_opt parse mismatch: {len(chunks)} 'Testing chunk size' "
                          f"markers vs {len(latencies)} '-> Latency' markers")
    return {c: lat for c, lat in zip(chunks, latencies) if math.isfinite(lat)}


def parse_all_metrics(stdout):
    values = parse_metrics(stdout)
    for key, regex in EXTRA_PATTERNS.items():
        m = re.search(regex, stdout)
        values[key] = int(m.group(1)) if m else None
    return values


def run_main(args_list):
    cmd = [PYTHON, "main.py"] + args_list + ["--display_plots", "false"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    return cmd, proc


def do_list():
    print(f"{'Figure':<8} {'Description'}")
    print("-" * 90)
    for key, fig in FIGURES.items():
        print(f"{key:<8} {fig['title']}")


def print_table(fig, results):
    for sl in fig["sls"]:
        print(f"\n{fig['arch_model'].upper()} SL={sl}")
        hdr = (f"{'Config':<12} {'Latency(ms)':>12} {'ACIM':>6} {'PMCA':>6} {'TOT':>5} "
               f"{'Throughput':>12} {'EnergyEff':>11}")
        print(hdr)
        print("-" * len(hdr))
        for label, _ in fig["sweep"]:
            m = results[sl][label]
            acim = m["acim_used"]
            pmca = m["pmca_red_used"]
            tot = (acim or 0) + (pmca or 0)
            print(f"{label:<12} {m['latency_ms']:>12} {acim:>6} {pmca:>6} {tot:>5} "
                  f"{m['throughput']:>12} {m['energy_eff']:>11}")

        throughputs = [results[sl][label]["throughput"] for label, _ in fig["sweep"]]
        effs = [results[sl][label]["energy_eff"] for label, _ in fig["sweep"]]
        print(f"  -> throughput swing: {max(throughputs) / min(throughputs):.4f}x    "
              f"energy-efficiency swing: {max(effs) / min(effs):.4f}x")


BAR_WIDTH = 0.55


def _style_axis(ax, labels):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.grid(axis="y", color="lightgray", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=11)


def _zoomed_ylim(values, pad_frac=0.6, top_frac=0.15):
    """Zoom the axis to where the bars actually differ, like the reference
    figure does -- a full 0-based range would flatten small swings."""
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        return 0, hi * 1.1 if hi else 1
    return max(0, lo - span * pad_frac), hi + span * top_frac


def save_plot(fig, results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [l for l, _ in fig["sweep"]]
    n_sl = len(fig["sls"])
    ncols = 2 * n_sl + 1
    fig_mpl, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 4.2))

    panel_letters = ["a", "b"]
    for i, sl in enumerate(fig["sls"]):
        throughputs = [results[sl][l]["throughput"] for l in labels]
        effs = [results[sl][l]["energy_eff"] for l in labels]
        letter = panel_letters[i] if i < len(panel_letters) else chr(ord("a") + i)

        ax = axes[2 * i]
        bars = ax.bar(range(len(labels)), throughputs, color="#8B1A1A", width=BAR_WIDTH,
                      edgecolor="black", linewidth=0.8, zorder=3)
        ax.set_ylim(0, max(throughputs) * 1.1)
        ax.set_ylabel("Throughput (Inf/s)", fontsize=12)
        ax.set_title(f"({letter}) SL={sl}", fontsize=13)
        _style_axis(ax, labels)

        ax = axes[2 * i + 1]
        bars = ax.bar(range(len(labels)), effs, color="#2E8B99", width=BAR_WIDTH,
                      edgecolor="black", linewidth=0.8, zorder=3)
        ax.set_ylim(*_zoomed_ylim(effs))
        ax.set_ylabel("Energy Efficiency (Inf/s/W)", fontsize=12)
        ax.set_title(f"({letter}) SL={sl}", fontsize=13)
        _style_axis(ax, labels)

    # ACIM/PMCA node usage depends only on opt_level, not SL -- one shared panel.
    sl0 = fig["sls"][0]
    acims = [results[sl0][l]["acim_used"] for l in labels]
    pmcas = [results[sl0][l]["pmca_red_used"] for l in labels]
    totals = [a + p for a, p in zip(acims, pmcas)]

    ax = axes[-1]
    xs = range(len(labels))
    bars_acim = ax.bar(xs, acims, color="#3B1F4A", width=BAR_WIDTH, label="ACIM",
                       edgecolor="black", linewidth=0.8, zorder=3)
    bars_pmca = ax.bar(xs, pmcas, bottom=acims, color="#F5D76E", width=BAR_WIDTH, label="PMCA + MACE",
                       edgecolor="black", linewidth=0.8, zorder=3)
    ymin, _ = _zoomed_ylim(totals)
    ax.set_ylim(ymin, 92)
    for x, a, p in zip(xs, acims, pmcas):
        ax.annotate(f"{a:.0f}", (x, max(ymin, 0) + (a - max(ymin, 0)) / 2),
                    ha="center", va="center", fontsize=11, color="white", fontweight="bold")
        if p >= 3:
            ax.annotate(f"{p:.0f}", (x, a + p / 2),
                        ha="center", va="center", fontsize=11, fontweight="bold")
        else:
            ax.annotate(f"{p:.0f}", (x, a + p),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Total number of used nodes", fontsize=12)
    ax.set_title("(c) Both SLs", fontsize=13)
    ax.legend(fontsize=11)
    _style_axis(ax, labels)

    fig_mpl.suptitle(fig["title"], fontweight="bold", fontsize=14)
    fig_mpl.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig_mpl.savefig(out_path, dpi=150)
    plt.close(fig_mpl)


def _args_with_sl(common_args, sl):
    """Return common_args with --sl overridden to `sl`."""
    args = list(common_args)
    if "--sl" in args:
        i = args.index("--sl")
        args[i + 1] = str(sl)
    else:
        args += ["--sl", str(sl)]
    return args


def _args_with_prefill(common_args, prefill):
    """Return common_args with --prefill_size overridden to `prefill`."""
    args = list(common_args)
    if "--prefill_size" in args:
        i = args.index("--prefill_size")
        args[i + 1] = str(prefill)
    else:
        args += ["--prefill_size", str(prefill)]
    return args


def print_table_model_sweep(fig, results):
    for sl in fig["sls"]:
        print(f"\nSL={sl}")
        for model in fig["models"]:
            print(f"  {fig['model_labels'].get(model, model)}")
            hdr = f"    {'Config':<12} {'Throughput':>12} {'EnergyEff':>11}"
            print(hdr)
            print("    " + "-" * (len(hdr) - 4))
            for label, _, _ in PIPE_CONFIGS:
                m = results[sl][model][label]
                print(f"    {label:<12} {m['throughput']:>12} {m['energy_eff']:>11}")
            serial = results[sl][model]["Serial"]
            xb = results[sl][model]["XB-pipe"]
            th_ratio = xb["throughput"] / serial["throughput"] if serial["throughput"] else float("nan")
            eff_ratio = xb["energy_eff"] / serial["energy_eff"] if serial["energy_eff"] else float("nan")
            print(f"    -> XB-pipe/Serial: throughput x{th_ratio:.3g}, energy-efficiency x{eff_ratio:.3g}")


def _plot_model_sweep_panel(ax, fig_spec, results_sl, metric_key, ylabel, ylim=None):
    models = fig_spec["models"]
    configs = [label for label, _, _ in PIPE_CONFIGS]
    n_cfg = len(configs)
    gap = fig_spec.get("model_gap", 1.2)
    bar_step = fig_spec.get("bar_step", 1.0)

    xs, colors, values = [], [], []
    model_spans = []
    idx = 0.0
    for model in models:
        shades = fig_spec["model_colors"][model]
        start = idx
        for j, label in enumerate(configs):
            xs.append(idx)
            colors.append(shades[j])
            values.append(results_sl[model][label][metric_key])
            idx += bar_step
        model_spans.append((model, start, idx - 1.0))
        idx += gap

    ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.6,
           width=fig_spec.get("bar_width", 0.85), zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(configs * len(models), rotation=90, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", color="lightgray", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if fig_spec.get("hide_top_right_spines"):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for model, start, end in model_spans:
        center = (start + end) / 2
        ax.text(center, fig_spec.get("model_label_dy", -0.30), fig_spec["model_labels"].get(model, model),
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=10, fontweight="bold")

    if fig_spec.get("group_label"):
        overall_center = (model_spans[0][1] + model_spans[-1][2]) / 2
        ax.text(overall_center, fig_spec.get("group_label_dy", -0.55), fig_spec["group_label"],
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=10, fontweight="bold")

    top = max(values) if values else 1.0
    ylim_final = ylim if ylim else (0, top * 1.3)
    # Cap label height to stay inside the axis -- otherwise, with a tight ylim (a bar
    # sitting close to the cap), "value*1.03" lands above the panel and collides with
    # whatever is drawn above it (title, next panel's label row).
    label_ceiling = ylim_final[1] * 0.97

    for model, start, _ in model_spans:
        serial_v = results_sl[model]["Serial"][metric_key]
        xb_v = results_sl[model]["XB-pipe"][metric_key]
        ratio = xb_v / serial_v if serial_v else float("nan")
        color = "green" if ratio >= 1 else "red"
        x_arrow = start
        ax.annotate("", xy=(x_arrow, xb_v), xytext=(x_arrow, serial_v),
                    arrowprops=dict(arrowstyle="-|>", color=color, linestyle="dashed", lw=1.3),
                    zorder=5)
        label_dx = fig_spec.get("arrow_label_dx", 0.0)
        label_y_mult_spec = fig_spec.get("arrow_label_y_mult", 1.03)
        label_y_mult = (label_y_mult_spec.get(metric_key, 1.03)
                        if isinstance(label_y_mult_spec, dict) else label_y_mult_spec)
        label_va = "center" if label_dx else "bottom"
        ax.text(x_arrow + label_dx, min(max(serial_v, xb_v) * label_y_mult, label_ceiling),
                f"x{ratio:.3g}", ha="center", va=label_va, fontsize=10, color=color, fontweight="bold")

    ax.set_ylim(*ylim_final)
    return model_spans


def save_plot_model_sweep(fig, results, out_path_template):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel_letters = ["a", "b"]
    saved = []
    for i, sl in enumerate(fig["sls"]):
        letter = panel_letters[i] if i < len(panel_letters) else chr(ord("a") + i)
        fig_mpl, (ax_th, ax_eff) = plt.subplots(1, 2, figsize=fig.get("figsize", (18, 5.5)))

        overrides = fig.get("ylim_overrides", {})
        _plot_model_sweep_panel(ax_th, fig, results[sl], "throughput", "Throughput (Inf/s)",
                                ylim=overrides.get((sl, "throughput")))
        _plot_model_sweep_panel(ax_eff, fig, results[sl], "energy_eff", "Energy Efficiency (Inf/s/W)",
                                ylim=overrides.get((sl, "energy_eff")))

        suptitle = fig.get("suptitle") or f"({letter}) SL={sl}"
        fig_mpl.suptitle(suptitle, fontweight="bold", fontsize=13)
        if fig.get("group_label"):
            # Fixed layout instead of tight_layout: tight_layout only accounts for the
            # auto-generated tick labels, not the extra manual model/group label rows
            # below them, so it under-reserves space and everything collapses together.
            top = 0.82 if "\n" in suptitle else 0.90
            fig_mpl.subplots_adjust(left=0.08, right=0.98, top=top, bottom=0.28, wspace=0.25)
        else:
            top = 0.85 if "\n" in suptitle else 0.95
            fig_mpl.tight_layout(rect=(0, 0.08, 1, top))

        out_path = out_path_template.format(letter=letter)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig_mpl.savefig(out_path, dpi=150)
        plt.close(fig_mpl)
        saved.append(out_path)
    return saved


def save_plot_model_sweep_combined(fig, results, out_path):
    """Combine a 2-SL model_sweep figure's (a)/(b) panels into one image with a single
    shared caption above, instead of two separate per-SL files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sls = fig["sls"]
    overrides = fig.get("ylim_overrides", {})

    # Each SL's pair of subplots gets its own full-width row -- with 5 models x 6
    # configs (30 bars) per subplot, a 4-across single row leaves far too little
    # width per bar and every label collides.
    fig_mpl, ((ax_a_th, ax_a_eff), (ax_b_th, ax_b_eff)) = plt.subplots(2, 2, figsize=(18, 11))
    _plot_model_sweep_panel(ax_a_th, fig, results[sls[0]], "throughput", "Throughput (Inf/s)",
                            ylim=overrides.get((sls[0], "throughput")))
    _plot_model_sweep_panel(ax_a_eff, fig, results[sls[0]], "energy_eff", "Energy Efficiency (Inf/s/W)",
                            ylim=overrides.get((sls[0], "energy_eff")))
    _plot_model_sweep_panel(ax_b_th, fig, results[sls[1]], "throughput", "Throughput (Inf/s)",
                            ylim=overrides.get((sls[1], "throughput")))
    _plot_model_sweep_panel(ax_b_eff, fig, results[sls[1]], "energy_eff", "Energy Efficiency (Inf/s/W)",
                            ylim=overrides.get((sls[1], "energy_eff")))

    fig_mpl.suptitle(fig["combined_caption"], fontweight="bold", fontsize=11)
    fig_mpl.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.13,
                            hspace=0.40, wspace=0.25)

    fig_mpl.canvas.draw()
    pos_a0, pos_a1 = ax_a_th.get_position(), ax_a_eff.get_position()
    pos_b0, pos_b1 = ax_b_th.get_position(), ax_b_eff.get_position()
    # Model names are drawn inside _plot_model_sweep_panel using the axes transform
    # (fig["model_label_dy"] below the axes bottom) -- find their actual figure-space
    # row so (a)/(b) land below THAT, not beside/above it.
    model_row_dy = fig.get("model_label_dy", -0.30)
    axes_height_a = pos_a0.y1 - pos_a0.y0
    axes_height_b = pos_b0.y1 - pos_b0.y0
    label_y_a = pos_a0.y0 + model_row_dy * axes_height_a - 0.025
    label_y_b = pos_b0.y0 + model_row_dy * axes_height_b - 0.025
    fig_mpl.text(0.5, label_y_a, "(a)", ha="center", fontsize=16, fontweight="bold")
    fig_mpl.text(0.5, label_y_b, "(b)", ha="center", fontsize=16, fontweight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig_mpl.savefig(out_path, dpi=150)
    plt.close(fig_mpl)
    return out_path


def _results_cache_path(key):
    return os.path.join(FIGURES_DIR, f"fig{key}_results.json")


def _save_results_cache(key, results):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    with open(_results_cache_path(key), "w") as f:
        json.dump(results, f, indent=2)


def _load_results_cache(key, top_keys):
    """Load a figure's cached results, keyed by `top_keys` (e.g. fig["sls"] or
    fig["bandwidths"]) -- JSON keys are always strings, so they're cast back to int."""
    path = _results_cache_path(key)
    with open(path) as f:
        raw = json.load(f)
    return {int(k): raw[str(k)] for k in top_keys}


TIMING_PATH = os.path.join(FIGURES_DIR, "run_timing.json")


def _load_timing():
    if not os.path.exists(TIMING_PATH):
        return {}
    with open(TIMING_PATH) as f:
        return json.load(f)


def _print_time_estimate(key, total_runs):
    """Print an ETA based on this figure's own run history, if any is recorded yet."""
    entry = _load_timing().get(key)
    if entry and entry.get("total_runs"):
        avg = entry["total_elapsed"] / entry["total_runs"]
        est = avg * total_runs
        print(f"[ESTIMATE] {total_runs} simulation(s) to run; based on {entry['total_runs']} "
              f"prior run(s) averaging {avg:.1f}s/run, expect ~{est / 60:.1f} min total.")
    else:
        print(f"[ESTIMATE] {total_runs} simulation(s) to run; no timing history yet for this "
              f"figure -- actual timing will be recorded and used for future estimates.")


def _record_timing(key, total_runs, total_elapsed):
    """Accumulate this run's timing into the figure's persistent history."""
    data = _load_timing()
    entry = data.setdefault(key, {"total_runs": 0, "total_elapsed": 0.0})
    entry["total_runs"] += total_runs
    entry["total_elapsed"] += total_elapsed
    os.makedirs(FIGURES_DIR, exist_ok=True)
    with open(TIMING_PATH, "w") as f:
        json.dump(data, f, indent=2)


def replot_figure_model_sweep(key):
    """Re-draw a model_sweep figure's plot from its cached results -- no simulation."""
    fig = FIGURES[key]
    path = _results_cache_path(key)
    if not os.path.exists(path):
        print(f"No cached results at {os.path.relpath(path, REPO_ROOT)}; "
              f"run without --replot first.", file=sys.stderr)
        sys.exit(2)
    results = _load_results_cache(key, fig["sls"])
    letter_part = "" if len(fig["sls"]) == 1 else "{letter}"
    out_path_template = os.path.join(FIGURES_DIR, f"fig{key}{letter_part}_reproduced.png")
    saved = save_plot_model_sweep(fig, results, out_path_template)
    for p in saved:
        print(f"Plot saved to {os.path.relpath(p, REPO_ROOT)}")

    if fig.get("combined_caption") and len(fig["sls"]) == 2:
        combined_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
        save_plot_model_sweep_combined(fig, results, combined_path)
        print(f"Plot saved to {os.path.relpath(combined_path, REPO_ROOT)}")


def replot_figure_opt_sweep(key):
    """Re-draw an opt_sweep figure's plot (e.g. Fig. 7) from its cached results -- no simulation."""
    fig = FIGURES[key]
    path = _results_cache_path(key)
    if not os.path.exists(path):
        print(f"No cached results at {os.path.relpath(path, REPO_ROOT)}; "
              f"run without --replot first.", file=sys.stderr)
        sys.exit(2)
    results = _load_results_cache(key, fig["sls"])
    out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
    save_plot(fig, results, out_path)
    print(f"Plot saved to {os.path.relpath(out_path, REPO_ROOT)}")


def run_figure_model_sweep(key, make_plot=True):
    fig = FIGURES[key]
    print(f"=== Reproducing {fig['title']} ===")

    backup = f"{CONFIG_PATH}.reproduce_backup.{os.getpid()}"
    shutil.copy2(CONFIG_PATH, backup)
    print(f"[ARCH] saved current config to {os.path.basename(backup)} (restored on exit)")

    total_runs = len(fig["models"]) * len(fig["sls"]) * len(PIPE_CONFIGS)
    _print_time_estimate(key, total_runs)
    done_runs = 0
    start_time = time.time()

    results = {sl: {model: {} for model in fig["models"]} for sl in fig["sls"]}
    try:
        for model in fig["models"]:
            suite = MODEL_SUITES[model]
            apply_model_arch(suite["arch_model"])

            for sl in fig["sls"]:
                common_args = _args_with_sl(suite["common_args"], sl)
                for label, parall_type, pipe_type in PIPE_CONFIGS:
                    pct = 100.0 * done_runs / total_runs
                    elapsed = time.time() - start_time
                    print(f"[RUN {done_runs + 1}/{total_runs} = {pct:.1f}%, elapsed {elapsed:.0f}s] "
                          f"{model} SL={sl} {label} ...", flush=True)
                    _write_status(f"{done_runs + 1}/{total_runs} ({pct:.1f}%) -- running "
                                  f"{model} SL={sl} {label}, elapsed {elapsed:.0f}s")
                    cmd, proc = run_config(common_args, parall_type, pipe_type)
                    done_runs += 1
                    if proc.returncode != 0:
                        print(f"  -> ERROR: process exited {proc.returncode}")
                        print(proc.stderr[-2000:])
                        sys.exit(1)
                    results[sl][model][label] = parse_metrics(proc.stdout)
    finally:
        shutil.copy2(backup, CONFIG_PATH)
        os.remove(backup)
        print(f"[ARCH] restored original nodes/accelerator_config.py")

    total_elapsed = time.time() - start_time
    print(f"\n[DONE] {total_runs} runs in {total_elapsed:.0f}s "
          f"({total_elapsed / total_runs:.1f}s/run avg)")
    _write_status(f"{total_runs}/{total_runs} (100.0%) -- done, total elapsed {total_elapsed:.0f}s")
    _record_timing(key, total_runs, total_elapsed)

    _save_results_cache(key, results)
    print_table_model_sweep(fig, results)

    if make_plot:
        letter_part = "" if len(fig["sls"]) == 1 else "{letter}"
        out_path_template = os.path.join(FIGURES_DIR, f"fig{key}{letter_part}_reproduced.png")
        saved = save_plot_model_sweep(fig, results, out_path_template)
        for p in saved:
            print(f"\nPlot saved to {os.path.relpath(p, REPO_ROOT)}")

        if fig.get("combined_caption") and len(fig["sls"]) == 2:
            combined_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
            save_plot_model_sweep_combined(fig, results, combined_path)
            print(f"Plot saved to {os.path.relpath(combined_path, REPO_ROOT)}")


def _pipe_config_for_label(label):
    for l, parall_type, pipe_type in PIPE_CONFIGS:
        if l == label:
            return parall_type, pipe_type
    raise KeyError(f"No PIPE_CONFIGS entry for label {label!r}")


def _apply_link_bandwidth(bw):
    """Directly patch LINK_CONFIG["bandwidth_b_per_cycle"] in nodes/accelerator_config.py --
    there's no CLI flag for it, so this reuses set_model_arch.py's own line-editing helper."""
    with open(CONFIG_PATH) as f:
        lines = f.readlines()
    _sma_set_value(lines, "LINK_CONFIG", "bandwidth_b_per_cycle", bw)
    with open(CONFIG_PATH, "w") as f:
        f.writelines(lines)


def _apply_sram_tiles(n):
    """Directly patch SRAM_TILE_CONFIG["num_tiles"] -- SRAM_TILE_CONFIG["size"] is fixed
    at 1 MiB/tile, so num_tiles == MiB. No CLI flag for it, same approach as bandwidth."""
    with open(CONFIG_PATH) as f:
        lines = f.readlines()
    _sma_set_value(lines, "SRAM_TILE_CONFIG", "num_tiles", n)
    with open(CONFIG_PATH, "w") as f:
        f.writelines(lines)


def replot_figure_bw_sweep(key):
    """Re-draw a bw_sweep figure's plot (e.g. Fig. 9) from its cached results -- no simulation."""
    fig = FIGURES[key]
    path = _results_cache_path(key)
    if not os.path.exists(path):
        print(f"No cached results at {os.path.relpath(path, REPO_ROOT)}; "
              f"run without --replot first.", file=sys.stderr)
        sys.exit(2)
    results = _load_results_cache(key, fig["bandwidths"])
    out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
    save_plot_bw_sweep(fig, results, out_path)
    print(f"Plot saved to {os.path.relpath(out_path, REPO_ROOT)}")


def run_figure_bw_sweep(key, make_plot=True):
    fig = FIGURES[key]
    print(f"=== Reproducing {fig['title']} ===")

    backup = f"{CONFIG_PATH}.reproduce_backup.{os.getpid()}"
    shutil.copy2(CONFIG_PATH, backup)
    print(f"[ARCH] saved current config to {os.path.basename(backup)} (restored on exit)")

    total_runs = len(fig["bandwidths"]) * len(fig["models"]) * len(fig["configs"])
    _print_time_estimate(key, total_runs)
    done_runs = 0
    start_time = time.time()

    results = {bw: {model: {} for model in fig["models"]} for bw in fig["bandwidths"]}
    try:
        for bw in fig["bandwidths"]:
            for model in fig["models"]:
                suite = MODEL_SUITES[model]
                # Re-applying the model arch resets LINK_CONFIG to its default (512) too,
                # so it must run before each bandwidth override, not just once per model.
                apply_model_arch(suite["arch_model"])
                _apply_link_bandwidth(bw)
                common_args = _args_with_sl(suite["common_args"], fig["sl"])

                for label in fig["configs"]:
                    parall_type, pipe_type = _pipe_config_for_label(label)
                    pct = 100.0 * done_runs / total_runs
                    elapsed = time.time() - start_time
                    print(f"[RUN {done_runs + 1}/{total_runs} = {pct:.1f}%, elapsed {elapsed:.0f}s] "
                          f"{model} BW={bw} {label} ...", flush=True)
                    _write_status(f"{done_runs + 1}/{total_runs} ({pct:.1f}%) -- running "
                                  f"{model} BW={bw} {label}, elapsed {elapsed:.0f}s")
                    cmd, proc = run_config(common_args, parall_type, pipe_type)
                    done_runs += 1
                    if proc.returncode != 0:
                        print(f"  -> ERROR: process exited {proc.returncode}")
                        print(proc.stderr[-2000:])
                        sys.exit(1)
                    results[bw][model][label] = parse_metrics(proc.stdout)
    finally:
        shutil.copy2(backup, CONFIG_PATH)
        os.remove(backup)
        print(f"[ARCH] restored original nodes/accelerator_config.py")

    total_elapsed = time.time() - start_time
    print(f"\n[DONE] {total_runs} runs in {total_elapsed:.0f}s "
          f"({total_elapsed / total_runs:.1f}s/run avg)")
    _write_status(f"{total_runs}/{total_runs} (100.0%) -- done, total elapsed {total_elapsed:.0f}s")
    _record_timing(key, total_runs, total_elapsed)

    _save_results_cache(key, results)
    print_table_bw_sweep(fig, results)

    if make_plot:
        out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
        save_plot_bw_sweep(fig, results, out_path)
        print(f"\nPlot saved to {os.path.relpath(out_path, REPO_ROOT)}")


def print_table_bw_sweep(fig, results):
    for bw in fig["bandwidths"]:
        print(f"\nBW={bw}")
        for model in fig["models"]:
            print(f"  {fig['model_labels'].get(model, model)}")
            hdr = f"    {'Config':<10} {'Throughput':>12} {'EnergyEff':>11}"
            print(hdr)
            print("    " + "-" * (len(hdr) - 4))
            for label in fig["configs"]:
                m = results[bw][model][label]
                print(f"    {label:<10} {m['throughput']:>12} {m['energy_eff']:>11}")


def save_plot_bw_sweep(fig, results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bandwidths = fig["bandwidths"]
    models = fig["models"]
    configs = fig["configs"]
    bw_gap = 1.3
    model_gap = 0.4

    def _plot_panel(ax, metric_key, ylabel, panel_letter, color_model_map=None):
        color_model_map = color_model_map or {m: m for m in models}
        xs, colors, values = [], [], []
        model_spans = []   # (bw, model, start, end)
        bw_spans = []       # (bw, start, end)
        idx = 0.0
        for bw in bandwidths:
            bw_start = idx
            for model in models:
                mstart = idx
                for label in configs:
                    xs.append(idx)
                    colors.append(fig["model_colors"][color_model_map[model]][label])
                    values.append(results[bw][model][label][metric_key])
                    idx += 1.0
                model_spans.append((bw, model, mstart, idx - 1.0))
                idx += model_gap
            bw_spans.append((bw, bw_start, idx - model_gap - 1.0))
            idx += bw_gap - model_gap

        ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.6, width=0.85, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(configs * len(models) * len(bandwidths), rotation=90, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", color="lightgray", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"({panel_letter})", fontsize=13, fontweight="bold")

        for bw, model, mstart, mend in model_spans:
            center = (mstart + mend) / 2
            ax.text(center, -0.22, fig["model_labels"].get(model, model),
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
        for bw, start, end in bw_spans:
            center = (start + end) / 2
            ax.text(center, -0.32, f"BW={bw}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=10, fontweight="bold")

        top = max(values) if values else 1.0
        for bw, model, mstart, _ in model_spans:
            serial_v = results[bw][model]["Serial"][metric_key]
            xb_v = results[bw][model]["XB-pipe"][metric_key]
            ratio = xb_v / serial_v if serial_v else float("nan")
            color = "green" if ratio >= 1 else "red"
            ax.annotate("", xy=(mstart, xb_v), xytext=(mstart, serial_v),
                        arrowprops=dict(arrowstyle="-|>", color=color, linestyle="dashed", lw=1.2),
                        zorder=5)
            ax.text(mstart, max(serial_v, xb_v) * 1.03, f"x{ratio:.3g}",
                    ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")

        ax.set_ylim(0, top * 1.3)

    fig_mpl, (ax_th, ax_eff) = plt.subplots(1, 2, figsize=(15, 6.5))
    _plot_panel(ax_th, "throughput", "Throughput (Inf/s)", "a")
    # BERT-B/BERT-L colors are swapped in the energy-efficiency panel only,
    # matching the paper's corrected figure.
    _plot_panel(ax_eff, "energy_eff", "Energy Efficiency (Inf/s/W)", "b",
                color_model_map={"BERT_B": "BERT_L", "BERT_L": "BERT_B"})

    fig_mpl.suptitle(fig["title"], fontweight="bold", fontsize=13)
    # Fixed layout instead of tight_layout: tight_layout only accounts for the
    # auto-generated tick labels, not the extra manual model/BW label rows below
    # them, so it under-reserves space and the rows collide.
    fig_mpl.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.32, wspace=0.25)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig_mpl.savefig(out_path, dpi=150)
    plt.close(fig_mpl)


def _load_chunk_opt_cache(key, sls, groups):
    """Like _load_results_cache, but chunk sizes (the innermost key, per group) also
    need casting back from JSON's string keys to int. `groups` is fig["models"] for a
    chunk_opt_sweep figure, or fig["pipe_groups"]'s labels for a chunk_opt_pipe_sweep one."""
    path = _results_cache_path(key)
    with open(path) as f:
        raw = json.load(f)
    return {
        int(sl): {g: {int(c): v for c, v in raw[str(sl)][g].items()} for g in groups}
        for sl in sls
    }


def print_table_chunk_opt(fig, results):
    for sl in fig["sls"]:
        print(f"\nSL={sl}")
        for model in fig["models"]:
            chunk_latencies = results[sl][model]
            best_chunk = min(chunk_latencies, key=chunk_latencies.get)
            print(f"  {fig['model_labels'].get(model, model)}  (best chunk={best_chunk}, "
                  f"latency={chunk_latencies[best_chunk]:.3f} ms)")
            for c in sorted(chunk_latencies):
                marker = "  <-- best" if c == best_chunk else ""
                print(f"    chunk={c:<4} latency={chunk_latencies[c]:.3f} ms{marker}")


def save_plot_chunk_opt(fig, results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_gap = 1.0
    panel_letters = ["a", "b"]

    def _plot_row(ax, sl, letter):
        chunks_for_sl = sorted(results[sl][fig["models"][0]].keys())
        xs, colors, values = [], [], []
        model_spans = []
        idx = 0.0
        for model in fig["models"]:
            chunk_latencies = results[sl][model]
            best_chunk = min(chunk_latencies, key=chunk_latencies.get)
            start = idx
            for c in chunks_for_sl:
                xs.append(idx)
                colors.append(fig["best_color"] if c == best_chunk else fig["model_colors"][model])
                values.append(chunk_latencies[c])
                idx += 1.0
            model_spans.append((model, best_chunk, start, idx - 1.0))
            idx += model_gap

        ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.6, width=0.85, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(c) for c in chunks_for_sl] * len(fig["models"]), fontsize=7)
        ax.set_ylabel("Latency (ms)", fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", color="lightgray", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"({letter}) SL={sl}", fontsize=13, fontweight="bold", pad=18)

        for model, best_chunk, start, end in model_spans:
            center = (start + end) / 2
            ax.text(center, -0.14, fig["model_labels"].get(model, model),
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10)

        top = max(values) if values else 1.0
        for model, best_chunk, start, _ in model_spans:
            chunk_latencies = results[sl][model]
            x_first = start
            x_best = start + chunks_for_sl.index(best_chunk)
            x_last = start + len(chunks_for_sl) - 1
            v_first = chunk_latencies[chunks_for_sl[0]]
            v_best = chunk_latencies[best_chunk]
            v_last = chunk_latencies[chunks_for_sl[-1]]

            ax.text(x_first, v_first, f"{v_first:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(x_last, v_last, f"{v_last:.2f}", ha="center", va="bottom", fontsize=8)
            # Inside the bar, not above it -- the arrows both start right at its top,
            # so a label there would collide with whichever arrow's tip lands nearby.
            ax.text(x_best, v_best / 2, f"{v_best:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold")

            # Both comparisons are drawn as vertical lines anchored at the yellow (best)
            # bar's x position -- offset apart so they don't sit on top of each other,
            # rather than diagonal/horizontal lines that would cross other bars. Their
            # tip labels can still collide when v_first and v_last happen to be close
            # (e.g. chunk=1 and the largest chunk land at similar latency) -- the offset
            # keeps that from happening.
            x_offset = 0.4
            # When chunk=1 and the largest chunk land at nearly the same latency, both
            # tip labels want to sit at the same height too -- nudge the blue one up a
            # bit further so the two texts don't overlap even with the x offset.
            close_heights = abs(v_last - v_first) < 0.08 * top
            if x_best != x_first:
                ratio = v_first / v_best if v_best else float("nan")
                x_green = x_best - x_offset
                ax.annotate("", xy=(x_green, v_best), xytext=(x_green, v_first),
                            arrowprops=dict(arrowstyle="-|>", color="#6B8E23",
                                            linestyle="dashed", lw=1.3), zorder=5)
                ax.text(x_green, v_first, f"x{ratio:.3g}", ha="center", va="bottom",
                        fontsize=9, color="#6B8E23", fontweight="bold")
            if x_best != x_last:
                ratio = v_last / v_best if v_best else float("nan")
                x_blue = x_best + x_offset
                y_blue_label = v_last + (0.05 * top if close_heights else 0)
                ax.annotate("", xy=(x_blue, v_best), xytext=(x_blue, v_last),
                            arrowprops=dict(arrowstyle="-|>", color="#1E4B6B", lw=1.3), zorder=5)
                ax.text(x_blue, y_blue_label, f"x{ratio:.3g}", ha="center", va="bottom",
                        fontsize=9, color="#1E4B6B", fontweight="bold")

        ylim_override = fig.get("ylim_overrides", {}).get(sl)
        ax.set_ylim(0, ylim_override if ylim_override else top * 1.3)

    fig_mpl, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(16, 11))
    _plot_row(ax_a, fig["sls"][0], panel_letters[0])
    _plot_row(ax_b, fig["sls"][1], panel_letters[1])

    ax_a.legend(handles=[
        plt.Line2D([0], [0], color="#6B8E23", linestyle="dashed", lw=1.3,
                   label="Speed-up against vector-wise"),
        plt.Line2D([0], [0], color="#1E4B6B", lw=1.3, label="Speed-up against largest chunk"),
    ], fontsize=9, loc="upper left")

    fig_mpl.suptitle(fig["title"], fontweight="bold", fontsize=13)
    fig_mpl.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig_mpl.savefig(out_path, dpi=150)
    plt.close(fig_mpl)


def replot_figure_chunk_opt(key):
    """Re-draw a chunk_opt_sweep figure's plot (e.g. Fig. 10) from cached results -- no simulation."""
    fig = FIGURES[key]
    path = _results_cache_path(key)
    if not os.path.exists(path):
        print(f"No cached results at {os.path.relpath(path, REPO_ROOT)}; "
              f"run without --replot first.", file=sys.stderr)
        sys.exit(2)
    results = _load_chunk_opt_cache(key, fig["sls"], fig["models"])
    out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
    save_plot_chunk_opt(fig, results, out_path)
    print(f"Plot saved to {os.path.relpath(out_path, REPO_ROOT)}")


def run_figure_chunk_opt_sweep(key, make_plot=True):
    fig = FIGURES[key]
    print(f"=== Reproducing {fig['title']} ===")

    backup = f"{CONFIG_PATH}.reproduce_backup.{os.getpid()}"
    shutil.copy2(CONFIG_PATH, backup)
    print(f"[ARCH] saved current config to {os.path.basename(backup)} (restored on exit)")

    n_runs_per_model_sl = 1 + len(fig["extra_chunks"])  # chunk_opt run + extra_chunks runs
    total_runs = len(fig["models"]) * len(fig["sls"]) * n_runs_per_model_sl
    _print_time_estimate(key, total_runs)
    done_runs = 0
    start_time = time.time()

    results = {sl: {model: {} for model in fig["models"]} for sl in fig["sls"]}
    try:
        for model in fig["models"]:
            suite = MODEL_SUITES[model]
            apply_model_arch(suite["arch_model"])

            for sl in fig["sls"]:
                base_args = ["--model", model, "--autoregressive", "false", "--sl", str(sl),
                             "--parall_type", "intra_layer", "--pipe_type", "inter_block"]
                cap = fig["display_chunk_cap"][sl]

                pct = 100.0 * done_runs / total_runs
                elapsed = time.time() - start_time
                print(f"[RUN {done_runs + 1}/{total_runs} = {pct:.1f}%, elapsed {elapsed:.0f}s] "
                      f"{model} SL={sl} chunk_opt=true ...", flush=True)
                _write_status(f"{done_runs + 1}/{total_runs} ({pct:.1f}%) -- running "
                              f"{model} SL={sl} chunk_opt=true, elapsed {elapsed:.0f}s")
                cmd, proc = run_main(base_args + ["--chunk_opt", "true"])
                done_runs += 1
                if proc.returncode != 0:
                    print(f"  -> ERROR: process exited {proc.returncode}")
                    print(proc.stderr[-2000:])
                    sys.exit(1)
                chunk_latencies = _parse_chunk_opt_latencies(proc.stdout)
                results[sl][model].update({c: lat for c, lat in chunk_latencies.items() if c <= cap})

                for extra_chunk in fig["extra_chunks"]:
                    pct = 100.0 * done_runs / total_runs
                    elapsed = time.time() - start_time
                    print(f"[RUN {done_runs + 1}/{total_runs} = {pct:.1f}%, elapsed {elapsed:.0f}s] "
                          f"{model} SL={sl} chunk={extra_chunk} ...", flush=True)
                    _write_status(f"{done_runs + 1}/{total_runs} ({pct:.1f}%) -- running "
                                  f"{model} SL={sl} chunk={extra_chunk}, elapsed {elapsed:.0f}s")
                    cmd, proc = run_main(base_args + ["--chunk_opt", "false",
                                                       "--chunk_size_d", str(extra_chunk)])
                    done_runs += 1
                    if proc.returncode != 0:
                        print(f"  -> ERROR: process exited {proc.returncode}")
                        print(proc.stderr[-2000:])
                        sys.exit(1)
                    results[sl][model][extra_chunk] = parse_metrics(proc.stdout)["latency_ms"]
    finally:
        shutil.copy2(backup, CONFIG_PATH)
        os.remove(backup)
        print(f"[ARCH] restored original nodes/accelerator_config.py")

    total_elapsed = time.time() - start_time
    print(f"\n[DONE] {total_runs} runs in {total_elapsed:.0f}s "
          f"({total_elapsed / total_runs:.1f}s/run avg)")
    _write_status(f"{total_runs}/{total_runs} (100.0%) -- done, total elapsed {total_elapsed:.0f}s")
    _record_timing(key, total_runs, total_elapsed)

    _save_results_cache(key, results)
    print_table_chunk_opt(fig, results)

    if make_plot:
        out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
        save_plot_chunk_opt(fig, results, out_path)
        print(f"\nPlot saved to {os.path.relpath(out_path, REPO_ROOT)}")


def print_table_chunk_opt_pipe(fig, results):
    group_labels = [label for label, _ in fig["pipe_groups"]]
    for sl in fig["sls"]:
        print(f"\nSL={sl}")
        for label in group_labels:
            chunk_latencies = results[sl][label]
            best_chunk = min(chunk_latencies, key=chunk_latencies.get)
            print(f"  {label}  (best chunk={best_chunk}, latency={chunk_latencies[best_chunk]:.3f} ms)")
            for c in sorted(chunk_latencies):
                marker = "  <-- best" if c == best_chunk else ""
                print(f"    chunk={c:<4} latency={chunk_latencies[c]:.3f} ms{marker}")


def save_plot_chunk_opt_pipe(fig, results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group_labels = [label for label, _ in fig["pipe_groups"]]
    group_gap = 1.0
    panel_letters = ["a", "b"]

    def _plot_panel(ax, sl, letter):
        chunks_for_sl = sorted(results[sl][group_labels[0]].keys())
        xs, colors, values = [], [], []
        group_spans = []
        idx = 0.0
        for label in group_labels:
            chunk_latencies = results[sl][label]
            best_chunk = min(chunk_latencies, key=chunk_latencies.get)
            start = idx
            for c in chunks_for_sl:
                xs.append(idx)
                colors.append(fig["best_color"] if c == best_chunk else fig["group_colors"][label])
                values.append(chunk_latencies[c])
                idx += 1.0
            group_spans.append((label, best_chunk, start, idx - 1.0))
            idx += group_gap

        ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.6, width=0.85, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(c) for c in chunks_for_sl] * len(group_labels), fontsize=7)
        ax.set_ylabel("Latency (ms)", fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", color="lightgray", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"({letter}) {fig['model']} SL={sl}", fontsize=13, fontweight="bold")

        for label, best_chunk, start, end in group_spans:
            center = (start + end) / 2
            ax.text(center, -0.14, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=10)

        top = max(values) if values else 1.0
        for label, best_chunk, start, _ in group_spans:
            chunk_latencies = results[sl][label]
            x_first = start
            x_best = start + chunks_for_sl.index(best_chunk)
            x_last = start + len(chunks_for_sl) - 1
            v_first = chunk_latencies[chunks_for_sl[0]]
            v_best = chunk_latencies[best_chunk]
            v_last = chunk_latencies[chunks_for_sl[-1]]

            ax.text(x_first, v_first, f"{v_first:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(x_last, v_last, f"{v_last:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(x_best, v_best / 2, f"{v_best:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold")

        ylim_override = fig.get("ylim_overrides", {}).get(sl)
        ax.set_ylim(0, ylim_override if ylim_override else top * 1.3)

    fig_mpl, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(20, 6.5))
    _plot_panel(ax_a, fig["sls"][0], panel_letters[0])
    _plot_panel(ax_b, fig["sls"][1], panel_letters[1])

    fig_mpl.suptitle(fig["title"], fontweight="bold", fontsize=13)
    fig_mpl.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.16, wspace=0.15)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig_mpl.savefig(out_path, dpi=150)
    plt.close(fig_mpl)


def replot_figure_chunk_opt_pipe(key):
    """Re-draw Fig. 11's plot from cached results -- no simulation."""
    fig = FIGURES[key]
    path = _results_cache_path(key)
    if not os.path.exists(path):
        print(f"No cached results at {os.path.relpath(path, REPO_ROOT)}; "
              f"run without --replot first.", file=sys.stderr)
        sys.exit(2)
    group_labels = [label for label, _ in fig["pipe_groups"]]
    results = _load_chunk_opt_cache(key, fig["sls"], group_labels)
    out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
    save_plot_chunk_opt_pipe(fig, results, out_path)
    print(f"Plot saved to {os.path.relpath(out_path, REPO_ROOT)}")


def run_figure_chunk_opt_pipe_sweep(key, make_plot=True):
    fig = FIGURES[key]
    print(f"=== Reproducing {fig['title']} ===")

    # Inter-block for MobileBERT is exactly what Fig. 10 already ran and cached --
    # reuse it instead of re-running.
    fig10_cache_path = _results_cache_path("10")
    with open(fig10_cache_path) as f:
        fig10_raw = json.load(f)
    # JSON keys are always strings -- cast chunk sizes back to int so they match the
    # int-keyed dicts the live runs below produce (otherwise plotting/lookups by int
    # chunk size fail against this group's dict specifically).
    reused = {sl: {"Inter-block": {int(c): v for c, v in fig10_raw[str(sl)][fig["model"]].items()}}
              for sl in fig["sls"]}
    reused_group = "Inter-block"
    new_groups = [(label, pt) for label, pt in fig["pipe_groups"] if label != reused_group]

    backup = f"{CONFIG_PATH}.reproduce_backup.{os.getpid()}"
    shutil.copy2(CONFIG_PATH, backup)
    print(f"[ARCH] saved current config to {os.path.basename(backup)} (restored on exit)")
    print(f"[REUSE] {reused_group} taken from Fig. 10's cached MobileBERT results "
          f"(no re-simulation)")

    n_runs_per_group_sl = 1 + len(fig["extra_chunks"])
    total_runs = len(new_groups) * len(fig["sls"]) * n_runs_per_group_sl
    _print_time_estimate(key, total_runs)
    done_runs = 0
    start_time = time.time()

    results = {sl: {label: {} for label, _ in fig["pipe_groups"]} for sl in fig["sls"]}
    for sl in fig["sls"]:
        results[sl][reused_group] = reused[sl][reused_group]

    try:
        suite = MODEL_SUITES[fig["model"]]
        apply_model_arch(suite["arch_model"])

        for label, pipe_type in new_groups:
            for sl in fig["sls"]:
                base_args = ["--model", fig["model"], "--autoregressive", "false", "--sl", str(sl),
                             "--parall_type", "intra_layer", "--pipe_type", pipe_type]
                cap = fig["display_chunk_cap"][sl]

                pct = 100.0 * done_runs / total_runs
                elapsed = time.time() - start_time
                print(f"[RUN {done_runs + 1}/{total_runs} = {pct:.1f}%, elapsed {elapsed:.0f}s] "
                      f"{fig['model']} {label} SL={sl} chunk_opt=true ...", flush=True)
                _write_status(f"{done_runs + 1}/{total_runs} ({pct:.1f}%) -- running "
                              f"{fig['model']} {label} SL={sl} chunk_opt=true, elapsed {elapsed:.0f}s")
                cmd, proc = run_main(base_args + ["--chunk_opt", "true"])
                done_runs += 1
                if proc.returncode != 0:
                    print(f"  -> ERROR: process exited {proc.returncode}")
                    print(proc.stderr[-2000:])
                    sys.exit(1)
                chunk_latencies = _parse_chunk_opt_latencies(proc.stdout)
                results[sl][label].update({c: lat for c, lat in chunk_latencies.items() if c <= cap})

                for extra_chunk in fig["extra_chunks"]:
                    pct = 100.0 * done_runs / total_runs
                    elapsed = time.time() - start_time
                    print(f"[RUN {done_runs + 1}/{total_runs} = {pct:.1f}%, elapsed {elapsed:.0f}s] "
                          f"{fig['model']} {label} SL={sl} chunk={extra_chunk} ...", flush=True)
                    _write_status(f"{done_runs + 1}/{total_runs} ({pct:.1f}%) -- running "
                                  f"{fig['model']} {label} SL={sl} chunk={extra_chunk}, "
                                  f"elapsed {elapsed:.0f}s")
                    cmd, proc = run_main(base_args + ["--chunk_opt", "false",
                                                       "--chunk_size_d", str(extra_chunk)])
                    done_runs += 1
                    if proc.returncode != 0:
                        print(f"  -> ERROR: process exited {proc.returncode}")
                        print(proc.stderr[-2000:])
                        sys.exit(1)
                    results[sl][label][extra_chunk] = parse_metrics(proc.stdout)["latency_ms"]
    finally:
        shutil.copy2(backup, CONFIG_PATH)
        os.remove(backup)
        print(f"[ARCH] restored original nodes/accelerator_config.py")

    total_elapsed = time.time() - start_time
    print(f"\n[DONE] {total_runs} runs in {total_elapsed:.0f}s "
          f"({total_elapsed / total_runs:.1f}s/run avg)" if total_runs else "\n[DONE]")
    _write_status(f"{total_runs}/{total_runs} (100.0%) -- done, total elapsed {total_elapsed:.0f}s")
    if total_runs:
        _record_timing(key, total_runs, total_elapsed)

    _save_results_cache(key, results)
    print_table_chunk_opt_pipe(fig, results)

    if make_plot:
        out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
        save_plot_chunk_opt_pipe(fig, results, out_path)
        print(f"\nPlot saved to {os.path.relpath(out_path, REPO_ROOT)}")


def print_table_sram_sweep(fig, results):
    print()
    hdr = f"{'SRAM':<8} {'Throughput':>12} {'EnergyEff':>11}"
    print(hdr)
    print("-" * len(hdr))
    for n in fig["sram_values"]:
        m = results[n]
        print(f"{n} MiB{'':<3} {m['throughput']:>12} {m['energy_eff']:>11}")


def save_plot_sram_sweep(fig, results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sram_values = fig["sram_values"]
    labels = [f"{n} MiB" for n in sram_values]
    xs = list(range(len(sram_values)))

    fig_mpl, (ax_th, ax_eff) = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, metric_key, ylabel, color in [
        (ax_th, "throughput", "Throughput (Inf/s)", fig["throughput_color"]),
        (ax_eff, "energy_eff", "Energy Efficiency (Inf/s/W)", fig["energy_eff_color"]),
    ]:
        values = [results[n][metric_key] for n in sram_values]
        ax.bar(xs, values, color=color, edgecolor="black", linewidth=0.6, width=0.7, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", color="lightgray", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        if fig.get("group_label"):
            ax.text(0.5, -0.32, fig["group_label"], transform=ax.transAxes,
                    ha="center", va="top", fontsize=10)

    fig_mpl.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.30, wspace=0.3)
    fig_mpl.text(0.5, 0.02, "(b)", ha="center", fontsize=20, fontweight="bold")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig_mpl.savefig(out_path, dpi=150)
    plt.close(fig_mpl)


def replot_figure_sram_sweep(key):
    """Re-draw a sram_sweep figure's plot (e.g. Fig. 12(b)) from cached results -- no simulation."""
    fig = FIGURES[key]
    path = _results_cache_path(key)
    if not os.path.exists(path):
        print(f"No cached results at {os.path.relpath(path, REPO_ROOT)}; "
              f"run without --replot first.", file=sys.stderr)
        sys.exit(2)
    results = _load_results_cache(key, fig["sram_values"])
    out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
    save_plot_sram_sweep(fig, results, out_path)
    print(f"Plot saved to {os.path.relpath(out_path, REPO_ROOT)}")


def run_figure_sram_sweep(key, make_plot=True):
    fig = FIGURES[key]
    print(f"=== Reproducing {fig['title']} ===")

    backup = f"{CONFIG_PATH}.reproduce_backup.{os.getpid()}"
    shutil.copy2(CONFIG_PATH, backup)
    print(f"[ARCH] saved current config to {os.path.basename(backup)} (restored on exit)")

    total_runs = len(fig["sram_values"])
    _print_time_estimate(key, total_runs)
    done_runs = 0
    start_time = time.time()

    results = {}
    try:
        suite = MODEL_SUITES[fig["model"]]
        apply_model_arch(suite["arch_model"])

        for n in fig["sram_values"]:
            _apply_sram_tiles(n)

            pct = 100.0 * done_runs / total_runs
            elapsed = time.time() - start_time
            print(f"[RUN {done_runs + 1}/{total_runs} = {pct:.1f}%, elapsed {elapsed:.0f}s] "
                  f"{fig['model']} SRAM={n}MiB ...", flush=True)
            _write_status(f"{done_runs + 1}/{total_runs} ({pct:.1f}%) -- running "
                          f"{fig['model']} SRAM={n}MiB, elapsed {elapsed:.0f}s")
            cmd, proc = run_main(suite["common_args"] + ["--parall_type", "intra_layer",
                                                          "--pipe_type", "inter_block"])
            done_runs += 1
            if proc.returncode != 0:
                print(f"  -> ERROR: process exited {proc.returncode}")
                print(proc.stderr[-2000:])
                sys.exit(1)
            results[n] = parse_metrics(proc.stdout)
    finally:
        shutil.copy2(backup, CONFIG_PATH)
        os.remove(backup)
        print(f"[ARCH] restored original nodes/accelerator_config.py")

    total_elapsed = time.time() - start_time
    print(f"\n[DONE] {total_runs} runs in {total_elapsed:.0f}s "
          f"({total_elapsed / total_runs:.1f}s/run avg)")
    _write_status(f"{total_runs}/{total_runs} (100.0%) -- done, total elapsed {total_elapsed:.0f}s")
    _record_timing(key, total_runs, total_elapsed)

    _save_results_cache(key, results)
    print_table_sram_sweep(fig, results)

    if make_plot:
        out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
        save_plot_sram_sweep(fig, results, out_path)
        print(f"\nPlot saved to {os.path.relpath(out_path, REPO_ROOT)}")


def combine_fig12(out_path):
    """Assemble Fig. 12(a) + Fig. 12(b) into one image with the paper's combined caption.
    Runs whichever of the two sub-figures' simulations haven't been done yet (caches
    missing), so a single `--figure 12` works from scratch -- no manual multi-step
    `--figure 12a` then `--figure 12b` needed first."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_a = FIGURES["12a"]
    fig_b = FIGURES["12b"]

    if not os.path.exists(_results_cache_path("12a")):
        print("[12] fig12a_results.json missing -- running Fig. 12(a)'s simulations first ...")
        run_figure_model_sweep("12a", make_plot=False)
    if not os.path.exists(_results_cache_path("12b")):
        print("[12] fig12b_results.json missing -- running Fig. 12(b)'s simulations first ...")
        run_figure_sram_sweep("12b", make_plot=False)

    results_a = _load_results_cache("12a", fig_a["sls"])[fig_a["sls"][0]]
    results_b = _load_results_cache("12b", fig_b["sram_values"])

    fig_mpl, axes = plt.subplots(1, 4, figsize=(22, 5.5),
                                 gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    ax_a_th, ax_a_eff, ax_b_th, ax_b_eff = axes

    overrides = fig_a.get("ylim_overrides", {})
    _plot_model_sweep_panel(ax_a_th, fig_a, results_a, "throughput", "Throughput (Inf/s)",
                            ylim=overrides.get((1, "throughput")))
    _plot_model_sweep_panel(ax_a_eff, fig_a, results_a, "energy_eff", "Energy Efficiency (Inf/s/W)",
                            ylim=overrides.get((1, "energy_eff")))

    sram_values = fig_b["sram_values"]
    labels_b = [f"{n} MiB" for n in sram_values]
    xs_b = list(range(len(sram_values)))
    for ax, metric_key, ylabel, color in [
        (ax_b_th, "throughput", "Throughput (Inf/s)", fig_b["throughput_color"]),
        (ax_b_eff, "energy_eff", "Energy Efficiency (Inf/s/W)", fig_b["energy_eff_color"]),
    ]:
        values = [results_b[n][metric_key] for n in sram_values]
        ax.bar(xs_b, values, color=color, edgecolor="black", linewidth=0.6, width=0.7, zorder=3)
        ax.set_xticks(xs_b)
        ax.set_xticklabels(labels_b, rotation=90, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", color="lightgray", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.text(0.5, -0.32, fig_b["group_label"], transform=ax.transAxes,
                ha="center", va="top", fontsize=10)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    caption = ("Fig. 12.   Single token generation: (a) different pipeline and parallelism "
               "techniques for GPT-2 Small and T5, (b) impact of on-chip SRAM capacity under "
               "XB-Pipe for GPT-2 Small.")
    fig_mpl.suptitle(caption, fontweight="bold", fontsize=13)
    fig_mpl.subplots_adjust(left=0.045, right=0.99, top=0.84, bottom=0.30, wspace=0.35)

    # (a)/(b) sub-labels, centered under each pair of panels, using their actual
    # rendered positions (set after subplots_adjust, so this reads the real layout).
    fig_mpl.canvas.draw()
    pos_a0, pos_a1 = ax_a_th.get_position(), ax_a_eff.get_position()
    pos_b0, pos_b1 = ax_b_th.get_position(), ax_b_eff.get_position()
    center_a = (pos_a0.x0 + pos_a1.x1) / 2
    center_b = (pos_b0.x0 + pos_b1.x1) / 2
    label_y = 0.02
    fig_mpl.text(center_a, label_y, "(a)", ha="center", fontsize=16, fontweight="bold")
    fig_mpl.text(center_b, label_y, "(b)", ha="center", fontsize=16, fontweight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig_mpl.savefig(out_path, dpi=150)
    plt.close(fig_mpl)


def print_table_token_sweep(fig, results):
    for model in fig["models"]:
        print(f"\n{fig['model_labels'].get(model, model)}")
        hdr = f"  {'#tokens':<8} {'Throughput':>12} {'EnergyEff':>11}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for n in fig["n_tokens"]:
            m = results[model].get(n) or {}
            th = m.get("throughput")
            ee = m.get("energy_eff")
            th_s = f"{th}" if th is not None else "inf"
            ee_s = f"{ee}" if ee is not None else "inf"
            print(f"  {n:<8} {th_s:>12} {ee_s:>11}")


def save_plot_token_sweep(fig, results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = fig["models"]
    n_tokens = fig["n_tokens"]
    n_per_model = len(n_tokens)
    model_gap = 1.0

    def _plot_panel(ax, metric_key, ylabel):
        xs, colors, values = [], [], []
        model_spans = []
        idx = 0.0
        for model in models:
            start = idx
            for n in n_tokens:
                xs.append(idx)
                colors.append(fig["model_colors"][model])
                v = (results[model].get(n) or {}).get(metric_key)
                values.append(v if v is not None else float("nan"))
                idx += 1.0
            model_spans.append((model, start, idx - 1.0))
            idx += model_gap

        ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.6, width=0.8, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(1, 2000)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(n) for n in n_tokens] * len(models), rotation=90, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", color="lightgray", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for model, start, end in model_spans:
            center = (start + end) / 2
            ax.text(center, -0.12, fig["group_label"], transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9)
            ax.text(center, -0.21, fig["model_labels"].get(model, model),
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=10, fontweight="bold")

    fig_mpl, (ax_th, ax_eff) = plt.subplots(1, 2, figsize=(16, 6))
    _plot_panel(ax_th, "throughput", "Throughput (Inf/s)")
    _plot_panel(ax_eff, "energy_eff", "Energy Efficiency (Inf/s/W)")

    caption = ("Fig. 13.   Throughput (a) and energy efficiency (b) for different number "
               "of generated tokens during autoregressive decoding under XB-Pipe.")
    fig_mpl.suptitle(caption, fontweight="bold", fontsize=13)
    fig_mpl.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.235, wspace=0.25)

    fig_mpl.canvas.draw()
    pos_th, pos_eff = ax_th.get_position(), ax_eff.get_position()
    label_y = 0.012
    fig_mpl.text((pos_th.x0 + pos_th.x1) / 2, label_y, "(a)", ha="center",
                fontsize=16, fontweight="bold")
    fig_mpl.text((pos_eff.x0 + pos_eff.x1) / 2, label_y, "(b)", ha="center",
                fontsize=16, fontweight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig_mpl.savefig(out_path, dpi=150)
    plt.close(fig_mpl)


def replot_figure_token_sweep(key):
    """Re-draw a token_sweep figure's plot (e.g. Fig. 13) from cached results -- no simulation."""
    fig = FIGURES[key]
    path = _results_cache_path(key)
    if not os.path.exists(path):
        print(f"No cached results at {os.path.relpath(path, REPO_ROOT)}; "
              f"run without --replot first.", file=sys.stderr)
        sys.exit(2)
    with open(path) as f:
        raw = json.load(f)
    results = {model: {int(n): raw[model][str(n)] for n in fig["n_tokens"]} for model in fig["models"]}
    out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
    save_plot_token_sweep(fig, results, out_path)
    print(f"Plot saved to {os.path.relpath(out_path, REPO_ROOT)}")


def run_figure_token_sweep(key, make_plot=True, resume=False):
    fig = FIGURES[key]
    print(f"=== Reproducing {fig['title']} ===")

    results = {model: {} for model in fig["models"]}
    cache_path = _results_cache_path(key)
    if resume and os.path.exists(cache_path):
        with open(cache_path) as f:
            raw = json.load(f)
        for model in fig["models"]:
            for n in fig["n_tokens"]:
                m = raw.get(model, {}).get(str(n))
                if m and m.get("latency_ms") is not None:
                    results[model][n] = m
        n_done = sum(len(v) for v in results.values())
        print(f"[RESUME] {n_done} already-successful runs loaded from cache")

    backup = f"{CONFIG_PATH}.reproduce_backup.{os.getpid()}"
    shutil.copy2(CONFIG_PATH, backup)
    print(f"[ARCH] saved current config to {os.path.basename(backup)} (restored on exit)")

    total_runs = len(fig["models"]) * len(fig["n_tokens"])
    _print_time_estimate(key, total_runs)
    done_runs = 0
    start_time = time.time()

    try:
        for model in fig["models"]:
            suite = MODEL_SUITES[model]
            apply_model_arch(suite["arch_model"])
            base_prefill = fig["base_prefill"][model]

            for n in fig["n_tokens"]:
                done_runs += 1
                if n in results[model]:
                    print(f"[SKIP {done_runs}/{total_runs}] {model} tokens={n} already have "
                          f"a result for", flush=True)
                    continue

                prefill = base_prefill + (n - 1)
                base_args = _args_with_prefill(suite["common_args"], prefill) + [
                    "--parall_type", "intra_layer", "--pipe_type", "inter_block"]

                pct = 100.0 * (done_runs - 1) / total_runs
                elapsed = time.time() - start_time
                print(f"[RUN {done_runs}/{total_runs} = {pct:.1f}%, elapsed {elapsed:.0f}s] "
                      f"{model} tokens={n} (prefill={prefill}) ...", flush=True)
                _write_status(f"{done_runs}/{total_runs} ({pct:.1f}%) -- running "
                              f"{model} tokens={n} prefill={prefill}, elapsed {elapsed:.0f}s")
                cmd, proc = run_main(base_args)
                if proc.returncode != 0:
                    print(f"  -> ERROR: process exited {proc.returncode}")
                    print(proc.stderr[-2000:])
                    sys.exit(1)
                metrics = parse_metrics(proc.stdout)

                if metrics.get("latency_ms") is None:
                    # KV state at this prefill overflows the digital accelerators'
                    # memory -> main.py reports "Total latency: inf" -- the README's
                    # documented fix is to retry the same point with flash attention.
                    print(f"  -> latency=inf, retrying with --flash_attention true ...", flush=True)
                    cmd, proc = run_main(base_args + ["--flash_attention", "true"])
                    if proc.returncode != 0:
                        print(f"  -> ERROR: process exited {proc.returncode}")
                        print(proc.stderr[-2000:])
                        sys.exit(1)
                    metrics = parse_metrics(proc.stdout)

                results[model][n] = metrics
                # Save incrementally so a crash partway through doesn't lose earlier runs.
                _save_results_cache(key, results)
    finally:
        shutil.copy2(backup, CONFIG_PATH)
        os.remove(backup)
        print(f"[ARCH] restored original nodes/accelerator_config.py")

    total_elapsed = time.time() - start_time
    print(f"\n[DONE] {total_runs} runs in {total_elapsed:.0f}s "
          f"({total_elapsed / total_runs:.1f}s/run avg)")
    _write_status(f"{total_runs}/{total_runs} (100.0%) -- done, total elapsed {total_elapsed:.0f}s")
    _record_timing(key, total_runs, total_elapsed)

    _save_results_cache(key, results)
    print_table_token_sweep(fig, results)

    if make_plot:
        out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
        save_plot_token_sweep(fig, results, out_path)
        print(f"\nPlot saved to {os.path.relpath(out_path, REPO_ROOT)}")


ANALOG_TILE_ROWS = 512
ANALOG_TILE_COLS = 512

ANALOG_TILE_RE = re.compile(r"Tile (\d+), Tier (\d+)\s*\n-+\s*\n((?:.+\n)+?)=+")
ANALOG_LAYER_RE = re.compile(r"(\S+)\s+rows\((\d+),\s*(\d+)\)\s+cols\((\d+),\s*(\d+)\)")


def _parse_analog_mapping_debug(path):
    """Parse outputs/AnalogLayerMapping/<model>_AnalogLayerMappingDebug_<opt_level>.txt
    (written by AnalogPool.debug_print) into [{tile_id, tier_id, blocks:[...]}, ...]."""
    with open(path) as f:
        text = f.read()
    mapping = []
    for tm in ANALOG_TILE_RE.finditer(text):
        tile_id, tier_id, body = int(tm.group(1)), int(tm.group(2)), tm.group(3)
        blocks = [
            {"layer": lm.group(1), "rows": [int(lm.group(2)), int(lm.group(3))],
             "cols": [int(lm.group(4)), int(lm.group(5))]}
            for lm in ANALOG_LAYER_RE.finditer(body)
        ]
        mapping.append({"tile_id": tile_id, "tier_id": tier_id, "blocks": blocks})
    return mapping


def print_table_analog_mapping(fig, results):
    for letter, opt_level in fig["opt_levels"]:
        mapping = results[opt_level]
        n_tiles = len(mapping)
        n_layers = sum(len(t["blocks"]) for t in mapping)
        print(f"\n({letter}) {opt_level}: {n_tiles} tile/tier slots used, {n_layers} layer placements")
        for t in mapping:
            layers = ", ".join(b["layer"] for b in t["blocks"])
            print(f"    Tile {t['tile_id']} Tier {t['tier_id']}: {layers}")


def _analog_block_label_color(layer_name):
    """'fc3_b1' -> label 'L3', layer index 3, block index 1."""
    fc_part, _, block_part = layer_name.partition("_b")
    layer_idx = int(fc_part[2:]) if fc_part.startswith("fc") and fc_part[2:].isdigit() else 0
    label = rf"$\mathrm{{L}}_{{{layer_idx}}}$" if fc_part.startswith("fc") else layer_name
    block_idx = int(block_part) if block_part.isdigit() else 0
    return label, layer_idx, block_idx


# Per-block, per-layer-index (L0..L5) shades -- darkest for L0, lightest for L5.
ANALOG_BLOCK_PALETTES = {
    0: ["#242a34", "#363f4e", "#4c586c", "#60708a", "#9ca8ba", "#dee2e7"],  # blues
    1: ["#711921", "#9a232d", "#c33c48", "#cd6f77", "#dea1a7", "#f5d9da"],  # reds
}


def _analog_layer_color(layer_idx, block_idx):
    palette = ANALOG_BLOCK_PALETTES.get(block_idx % len(ANALOG_BLOCK_PALETTES))
    return palette[layer_idx % len(palette)]


def save_plot_analog_mapping(fig, results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors

    tile_gap = 0.15
    tile_step = ANALOG_TILE_COLS * (1 + tile_gap)

    def _row_width(opt_level):
        n = len(results[opt_level])
        return n * tile_step - tile_gap * ANALOG_TILE_COLS if n else 0.0

    # Every row shares the same x-axis span (the widest row's width) so tiles
    # in (a)/(b)/(c) start flush at the same horizontal position instead of
    # each subplot's equal-aspect box independently centering its own content.
    max_width = max((_row_width(ol) for _, ol in fig["opt_levels"]), default=ANALOG_TILE_COLS)

    def _plot_row(ax, opt_level, letter):
        mapping = results[opt_level]
        x = 0.0
        for t in mapping:
            # Every physical ACIM tile is a fixed ANALOG_TILE_ROWS x ANALOG_TILE_COLS
            # (512x512) crossbar -- draw the full tile footprint, not just the
            # bounding box of what's actually used within it, so box size reflects
            # real hardware geometry and unused capacity is visible as empty space.
            w, h = ANALOG_TILE_COLS, ANALOG_TILE_ROWS
            ax.add_patch(patches.Rectangle((x, 0), w, h, linewidth=1.2,
                                           edgecolor="black", facecolor="none", zorder=3))
            for b in t["blocks"]:
                label, layer_idx, block_idx = _analog_block_label_color(b["layer"])
                color = _analog_layer_color(layer_idx, block_idx)
                r0, r1 = b["rows"]
                c0, c1 = b["cols"]
                ax.add_patch(patches.Rectangle((x + c0, r0), c1 - c0, r1 - r0, linewidth=0.8,
                                               edgecolor="black", facecolor=color, zorder=4))
                # Palette runs dark (L0) to light (L5) -- pick readable text color per shade.
                rgb = mcolors.to_rgb(color)
                text_color = "black" if sum(rgb) > 1.5 else "white"
                ax.text(x + (c0 + c1) / 2, (r0 + r1) / 2, label, ha="center", va="center",
                        fontsize=17, color=text_color, fontweight="bold", zorder=5)
            x += tile_step

        ax.set_xlim(-0.02 * max_width, max_width * 1.02)
        ax.set_ylim(ANALOG_TILE_ROWS * 1.05, -ANALOG_TILE_ROWS * 0.05)  # inverted, rows grow downward
        ax.set_aspect("equal")
        ax.set_anchor("N")  # pack equal-aspect content to the top of its row slot
        ax.axis("off")
        # Row label sits to the left of the tiles, vertically centered --
        # the opt_level name itself is already spelled out in the suptitle.
        ax.text(-0.03, 0.5, f"({letter})", transform=ax.transAxes, ha="right", va="center",
                fontsize=18, fontweight="bold")

    n_rows = len(fig["opt_levels"])
    fig_mpl, axes = plt.subplots(n_rows, 1, figsize=(14, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    for ax, (letter, opt_level) in zip(axes, fig["opt_levels"]):
        _plot_row(ax, opt_level, letter)

    n_blocks = max((_analog_block_label_color(b["layer"])[2]
                    for mapping in results.values() for t in mapping for b in t["blocks"]),
                   default=0) + 1
    n_layers = max((_analog_block_label_color(b["layer"])[1]
                    for mapping in results.values() for t in mapping for b in t["blocks"]),
                   default=-1) + 1

    # Layer -> (size string, input-dependency group), fixed by the dumpModel
    # definition (models/model_trial.py): L0-L2 have no dependency (group 0),
    # L3-L4 depend on L0-L2 (group 1), L5 depends on L3-L4 (group 2).
    layer_info = {0: ("512x128", 0), 1: ("512x128", 0), 2: ("512x128", 0),
                  3: ("128x128", 1), 4: ("128x128", 1), 5: ("640x128", 2)}

    fig_mpl.suptitle(fig["title"], fontweight="bold", fontsize=13, y=0.995)
    fig_mpl.tight_layout(rect=(0, 0, 1, 0.99))
    fig_mpl.canvas.draw()  # finalize axes positions before measuring them below

    # Place the legend in the empty area to the right of rows (b)/(c) -- i.e.
    # below where the last two tiles of row (a) sit -- rather than floating it
    # outside the plot area, since that space is otherwise unused whitespace.
    ax_a = axes[0]
    n_tiles_a = len(results[fig["opt_levels"][0][1]])
    x_start_data = max(0, n_tiles_a - 2) * tile_step
    (x0_disp, _), (x1_disp, _) = ax_a.transData.transform([(x_start_data, 0), (max_width, 0)])
    x0, x1 = fig_mpl.transFigure.inverted().transform([(x0_disp, 0), (x1_disp, 0)])[:, 0]
    y_top = axes[1].get_position().y1 if n_rows > 1 else axes[0].get_position().y1
    y_bottom = axes[-1].get_position().y0
    legend_ax = fig_mpl.add_axes((x0, y_bottom, x1 - x0, y_top - y_bottom))
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")

    LEGEND_FONTSIZE = 12

    # Block0/Block1 shown as a row of small colored squares, one per layer
    # shade (dark L0 -> light L5), instead of a continuous gradient bar.
    sq_size = 0.05
    sq_gap = 0.012
    sq_x0 = 0.26
    bar_ys = [0.93 - blk * 0.16 for blk in range(n_blocks)]
    for blk, bar_y in zip(range(n_blocks), bar_ys):
        legend_ax.text(0.0, bar_y, f"Block{blk}", ha="left", va="center",
                       fontsize=LEGEND_FONTSIZE + 6, fontweight="bold")
        for i in range(n_layers):
            color = _analog_layer_color(i, blk)
            sx = sq_x0 + i * (sq_size + sq_gap)
            legend_ax.add_patch(patches.Rectangle((sx, bar_y - sq_size / 2), sq_size, sq_size,
                                                   facecolor=color, edgecolor="black",
                                                   linewidth=0.8))

    # Layer size / input-dependency-group text, e.g. "L0 -> size=(512x128), IN=I0".
    text_top = min(bar_ys, default=0.93) - sq_size / 2 - 0.16
    text_lines = [rf"$\mathrm{{L}}_{{{i}}}$ -> size=({layer_info[i][0]}), "
                 rf"IN=$\mathrm{{I}}_{{{layer_info[i][1]}}}$"
                 for i in range(n_layers)]
    legend_ax.text(0.12, text_top, "\n".join(text_lines), ha="left", va="top",
                   fontsize=LEGEND_FONTSIZE + 7, linespacing=1.9)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig_mpl.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig_mpl)


def replot_figure_analog_mapping(key):
    """Re-draw Fig. 6's plot from cached mapping data -- no simulation."""
    fig = FIGURES[key]
    path = _results_cache_path(key)
    if not os.path.exists(path):
        print(f"No cached results at {os.path.relpath(path, REPO_ROOT)}; "
              f"run without --replot first.", file=sys.stderr)
        sys.exit(2)
    with open(path) as f:
        results = json.load(f)  # {opt_level: [tile entries]}
    out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
    save_plot_analog_mapping(fig, results, out_path)
    print(f"Plot saved to {os.path.relpath(out_path, REPO_ROOT)}")


def run_figure_analog_mapping(key, make_plot=True):
    fig = FIGURES[key]
    print(f"=== Reproducing {fig['title']} ===")

    backup = f"{CONFIG_PATH}.reproduce_backup.{os.getpid()}"
    shutil.copy2(CONFIG_PATH, backup)
    print(f"[ARCH] saved current config to {os.path.basename(backup)} (restored on exit)")

    total_runs = len(fig["opt_levels"])
    _print_time_estimate(key, total_runs)
    done_runs = 0
    start_time = time.time()

    results = {}
    try:
        apply_model_arch(fig["arch_model"])

        for letter, opt_level in fig["opt_levels"]:
            pct = 100.0 * done_runs / total_runs
            elapsed = time.time() - start_time
            print(f"[RUN {done_runs + 1}/{total_runs} = {pct:.1f}%, elapsed {elapsed:.0f}s] "
                  f"({letter}) opt_level={opt_level} ...", flush=True)
            _write_status(f"{done_runs + 1}/{total_runs} ({pct:.1f}%) -- running "
                          f"({letter}) opt_level={opt_level}, elapsed {elapsed:.0f}s")
            args = ["--model", fig["model"], "--sl", "1", "--parall_type", "none",
                   "--pipe_type", "none", "--opt_level", opt_level]
            cmd, proc = run_main(args)
            done_runs += 1
            if proc.returncode != 0:
                print(f"  -> ERROR: process exited {proc.returncode}")
                print(proc.stderr[-2000:])
                sys.exit(1)
            debug_path = os.path.join(
                REPO_ROOT, "outputs", "AnalogLayerMapping",
                f"{fig['model']}_AnalogLayerMappingDebug_{opt_level}.txt")
            results[opt_level] = _parse_analog_mapping_debug(debug_path)
    finally:
        shutil.copy2(backup, CONFIG_PATH)
        os.remove(backup)
        print(f"[ARCH] restored original nodes/accelerator_config.py")

    total_elapsed = time.time() - start_time
    print(f"\n[DONE] {total_runs} runs in {total_elapsed:.0f}s "
          f"({total_elapsed / total_runs:.1f}s/run avg)")
    _write_status(f"{total_runs}/{total_runs} (100.0%) -- done, total elapsed {total_elapsed:.0f}s")
    _record_timing(key, total_runs, total_elapsed)

    _save_results_cache(key, results)
    print_table_analog_mapping(fig, results)

    if make_plot:
        out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
        save_plot_analog_mapping(fig, results, out_path)
        print(f"\nPlot saved to {os.path.relpath(out_path, REPO_ROOT)}")


def run_figure(key, make_plot=True, resume=False):
    fig = FIGURES[key]
    if fig.get("kind") == "analog_mapping":
        run_figure_analog_mapping(key, make_plot)
        return
    if fig.get("kind") == "model_sweep":
        run_figure_model_sweep(key, make_plot)
        return
    if fig.get("kind") == "sram_sweep":
        run_figure_sram_sweep(key, make_plot)
        return
    if fig.get("kind") == "token_sweep":
        run_figure_token_sweep(key, make_plot, resume=resume)
        return
    if fig.get("kind") == "chunk_opt_sweep":
        run_figure_chunk_opt_sweep(key, make_plot)
        return
    if fig.get("kind") == "chunk_opt_pipe_sweep":
        run_figure_chunk_opt_pipe_sweep(key, make_plot)
        return
    if fig.get("kind") == "bw_sweep":
        run_figure_bw_sweep(key, make_plot)
        return

    print(f"=== Reproducing {fig['title']} ===")

    backup = f"{CONFIG_PATH}.reproduce_backup.{os.getpid()}"
    shutil.copy2(CONFIG_PATH, backup)
    print(f"[ARCH] saved current config to {os.path.basename(backup)} (restored on exit)")

    total_runs = len(fig["sls"]) * len(fig["sweep"])
    _print_time_estimate(key, total_runs)
    done_runs = 0
    start_time = time.time()

    results = {sl: {} for sl in fig["sls"]}
    try:
        apply_model_arch(fig["arch_model"])

        for sl in fig["sls"]:
            for label, sweep_val in fig["sweep"]:
                pct = 100.0 * done_runs / total_runs
                elapsed = time.time() - start_time
                print(f"[RUN {done_runs + 1}/{total_runs} = {pct:.1f}%, elapsed {elapsed:.0f}s] "
                      f"SL={sl} {label} ({fig['sweep_flag']}={sweep_val}) ...", flush=True)
                _write_status(f"{done_runs + 1}/{total_runs} ({pct:.1f}%) -- running "
                              f"SL={sl} {label}, elapsed {elapsed:.0f}s")
                args = fig["fixed_args"] + ["--sl", str(sl), fig["sweep_flag"], sweep_val]
                cmd, proc = run_main(args)
                done_runs += 1
                if proc.returncode != 0:
                    print(f"  -> ERROR: process exited {proc.returncode}")
                    print(proc.stderr[-2000:])
                    sys.exit(1)
                results[sl][label] = parse_all_metrics(proc.stdout)
    finally:
        shutil.copy2(backup, CONFIG_PATH)
        os.remove(backup)
        print(f"[ARCH] restored original nodes/accelerator_config.py")

    total_elapsed = time.time() - start_time
    print(f"\n[DONE] {total_runs} runs in {total_elapsed:.0f}s "
          f"({total_elapsed / total_runs:.1f}s/run avg)")
    _write_status(f"{total_runs}/{total_runs} (100.0%) -- done, total elapsed {total_elapsed:.0f}s")
    _record_timing(key, total_runs, total_elapsed)

    _save_results_cache(key, results)
    print_table(fig, results)

    if make_plot:
        out_path = os.path.join(FIGURES_DIR, f"fig{key}_reproduced.png")
        save_plot(fig, results, out_path)
        print(f"\nPlot saved to {os.path.relpath(out_path, REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figure", type=str, default=None,
                        help=f"Figure to reproduce. Choices: {', '.join(FIGURES)}")
    parser.add_argument("--list", action="store_true",
                        help="List available figures and exit.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip saving the PNG plot; print the table only.")
    parser.add_argument("--replot", action="store_true",
                        help="Re-draw the plot from cached results (model_sweep figures only) "
                             "without re-running any simulation. Use after styling-only changes.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs already present in the cache with a non-null result "
                             "(token_sweep figures only). Use to continue after an interruption "
                             "or to only fill in failed/missing points.")
    args = parser.parse_args()

    if args.list or not args.figure:
        do_list()
        return

    if args.figure not in FIGURES:
        print(f"Unknown figure '{args.figure}'. Choices: {', '.join(FIGURES)}")
        sys.exit(2)

    kind = FIGURES[args.figure].get("kind")
    if kind == "combine":
        # Never simulates anything -- just (re-)assembles from other figures' caches.
        out_path = os.path.join(FIGURES_DIR, f"fig{args.figure}_reproduced.png")
        combine_fig12(out_path)
        print(f"Plot saved to {os.path.relpath(out_path, REPO_ROOT)}")
        return

    if args.replot:
        kind = FIGURES[args.figure].get("kind")
        if kind == "analog_mapping":
            replot_figure_analog_mapping(args.figure)
        elif kind == "model_sweep":
            replot_figure_model_sweep(args.figure)
        elif kind == "bw_sweep":
            replot_figure_bw_sweep(args.figure)
        elif kind == "chunk_opt_sweep":
            replot_figure_chunk_opt(args.figure)
        elif kind == "chunk_opt_pipe_sweep":
            replot_figure_chunk_opt_pipe(args.figure)
        elif kind == "sram_sweep":
            replot_figure_sram_sweep(args.figure)
        elif kind == "token_sweep":
            replot_figure_token_sweep(args.figure)
        else:
            replot_figure_opt_sweep(args.figure)
        return

    run_figure(args.figure, make_plot=not args.no_plot, resume=args.resume)


if __name__ == "__main__":
    main()
