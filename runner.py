#!/usr/bin/env python3
#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""
sweep_runner.py
Run a grid-search of model / pipeline options and log the metrics that the
target script prints to stdout.

Usage
-----
$ python sweep_runner.py
"""

import csv
import pathlib
import subprocess
import re
from datetime import datetime
from pathlib import Path


OUT_PATH = "outputs/Sweeps"
Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
# -------- 1) Parameter grid --------------------------------------------------
MODELS = ["MobileBERT"] #["MobileBERT", "BERT_B", "BERT_L", "NanoGPT", GPT2_small", "T5_encoder", "T5_decoder"]
PREFILL = 512 
SL = [128] #[128,384]
AUTOREGRESSIVE = False #["False", "True"] #False for encoder and decoder (prefill), True for decoder in autoregressive mode


# -------- 2) Regex patterns for the numbers you care about -------------------
PATTERNS = {
    "total_latency_ms"     : r"Total latency:\s+([\d.]+)\s*ms",
    "analog_energy_mJ"     : r"Analog energy:\s+([\d.]+)\s*mJ",
    "digital_energy_mJ"    : r"Digital Accelerators energy:\s+([\d.]+)\s*mJ",
    "total_energy_mJ"      : r"Total energy:\s+([\d.]+)\s*mJ",
    "avg_power_W"          : r"Total avg power:\s+([\d.]+)\s*W",
    "area_mm2"             : r"Total area:\s+([\d.]+)\s*mm",
    "TDP_W"                : r"TDP:\s+([\d.]+)\s*W",
    "throughput_inf_s"     : r"Throughput:\s+([\d.]+)\s*Inf/s",
    "energy_eff_inf_s_W"   : r"Energy efficiency:\s+([\d.]+)\s*Inf/s/W",
    "execution_time_s"     : r"Execution time:\s+([\d.]+)\s*seconds?",
}

# -------- 3) Output CSV path --------------------------------------------------
models_list = "_".join(MODELS)
out_file = pathlib.Path(f"{OUT_PATH}/run_metrics_{models_list}_SL{SL}" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")

with out_file.open("w", newline="") as csv_fh:
    fieldnames = (
        ["timestamp", "cmd"] +
        list(PATTERNS.keys()) +
        ["model", "parall_type", "pipe_type", "chunk_size_d", "chunk_opt", "sl", "autoregressive", "prefill_size"]
    )
    writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    writer.writeheader()

    # -------- 4) Sweep loop ---------------------------------------------------
    for sl in SL :
        CONFIGS = [
            # Each dict becomes one set of CLI options
            {"parall_type": "none",        "pipe_type": "none",               "chunk_size_d": 8, "sl":sl, "autoregressive":AUTOREGRESSIVE, "prefill_size":PREFILL},
            {"parall_type": "intra_layer", "pipe_type": "none",               "chunk_size_d": 8, "sl":sl, "autoregressive":AUTOREGRESSIVE, "prefill_size":PREFILL},
            {"parall_type": "intra_layer", "pipe_type": "intra_layer",        "chunk_size_d": 8, "sl":sl, "autoregressive":AUTOREGRESSIVE, "prefill_size":PREFILL},
            {"parall_type": "intra_layer", "pipe_type": "inter_layer",        "chunk_size_d": 8, "sl":sl, "autoregressive":AUTOREGRESSIVE, "prefill_size":PREFILL},
            {"parall_type": "intra_layer", "pipe_type": "inter_intra_layer",  "chunk_size_d": 8, "sl":sl, "autoregressive":AUTOREGRESSIVE, "prefill_size":PREFILL},
            {"parall_type": "intra_layer", "pipe_type": "inter_block",        "chunk_size_d": 8, "sl":sl, "autoregressive":AUTOREGRESSIVE, "prefill_size":PREFILL},
            # {"parall_type": "intra_layer", "pipe_type": "inter_block",        "chunk_size_d": 4, "sl":sl,  "autoregressive":AUTOREGRESSIVE, "prefill_size":PREFILL},
            # {"parall_type": "intra_layer", "pipe_type": "inter_block",        "chunk_size_d": 1, "sl":sl, "autoregressive":AUTOREGRESSIVE, "prefill_size":PREFILL},
            # #last row has chunk_opt instead of chunk_size_d
            # {"parall_type": "intra_layer", "pipe_type": "inter_block",        "chunk_opt": True, "sl":sl, "autoregressive":AUTOREGRESSIVE, "prefill_size":PREFILL},
            # {"parall_type": "intra_layer", "pipe_type": "inter_intra_layer",        "chunk_opt": True, "sl":sl},
            # {"parall_type": "intra_layer", "pipe_type": "inter_intra_layer",        "chunk_size_d": 4, "sl":sl},
            # {"parall_type": "intra_layer", "pipe_type": "inter_intra_layer",        "chunk_size_d": 1, "sl":sl},
            # {"parall_type": "intra_layer", "pipe_type": "inter_layer",        "chunk_opt": True, "sl":sl},
            # {"parall_type": "intra_layer", "pipe_type": "inter_layer",        "chunk_size_d": 4, "sl":sl},
            # {"parall_type": "intra_layer", "pipe_type": "inter_layer",        "chunk_size_d": 1, "sl":sl},

        ]
        for model in MODELS:
            for cfg in CONFIGS:
                # Build CLI
                cmd = ["/opt/conda/envs/sda/bin/python", "main.py", "--model", model]  # <- replace run_model.py if needed
                for k, v in cfg.items():
                    flag = f"--{k}"
                    value = str(v) if not isinstance(v, bool) else str(v).lower()
                    # boolean flags become "--flag True/False"
                    cmd += [flag, value]

                # Run
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False  # keep going even if the script exits non-zero
                )
                stdout = proc.stdout

                # Parse metrics
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "cmd": " ".join(cmd),
                    "model": model,
                    **{k: cfg.get(k, "") for k in ("parall_type", "pipe_type", "chunk_size_d", "chunk_opt", "sl", "autoregressive", "prefill_size")}
                }

                for key, regex in PATTERNS.items():
                    m = re.search(regex, stdout)
                    row[key] = float(m.group(1)) if m else ""

                writer.writerow(row)

                # Console feedback
                model_tag = f"{model}/SL{sl}/{cfg['parall_type']}/{cfg.get('pipe_type','')}"
                print(f"Finished {model_tag:40s} | total latency {row['total_latency_ms']} ms")

print(f"\nDone! Results saved to {out_file.resolve()}")
