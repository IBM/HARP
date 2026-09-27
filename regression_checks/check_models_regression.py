#!/usr/bin/env python3
#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""
check_models_regression.py

Regression check across models and pipeline configurations, comparing
Total latency against known-good reference values.

For each model suite the TABLE III architecture is applied first (via
set_model_arch.py), then every pipeline configuration is run and its latency
compared against the reference. Only latency is asserted; throughput and energy
are recorded and printed but never treated as failures.

Reference values that are not yet established are marked None -- those configs
still run and report actual numbers, but never fail.

WARNING: this rewrites nodes/accelerator_config.py as it goes. The original
contents are restored on exit unless --keep-arch is passed.

Usage
-----
$ python regression_checks/check_models_regression.py
$ python regression_checks/check_models_regression.py --list
$ python regression_checks/check_models_regression.py --models MobileBERT
$ python regression_checks/check_models_regression.py --models BERT_B,T5_encoder --tol 1.0
$ python regression_checks/check_models_regression.py --models T5_decoder --record
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

from colorama import init as colorama_init
from colorama import Fore, Style

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from set_model_arch import MODEL_ARCH, resolve as resolve_arch_model

colorama_init()


def _color(text, color):
    return f"{color}{Style.BRIGHT}{text}{Style.RESET_ALL}"


def _status_text(status, width=None):
    """Render a status word: green+bold PASS, red+bold FAIL/ERROR, orange+bold PENDING."""
    padded = f"{status:<{width}}" if width else status
    if status == "PASS":
        return _color(padded, Fore.GREEN)
    if status in ("FAIL", "ERROR"):
        return _color(padded, Fore.RED)
    if status == "PENDING":
        return _color(padded, Fore.YELLOW)
    return padded


PYTHON = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SET_MODEL_ARCH_PATH = os.path.join(HERE, "set_model_arch.py")
CONFIG_PATH = os.path.join(REPO_ROOT, "nodes", "accelerator_config.py")
STATUS_FILE = os.path.join(HERE, "regression_status.txt")

PATTERNS = {
    "latency_ms":        r"Total latency:\s+([\d.]+)\s*ms",
    "throughput":        r"Throughput:\s+([\d.]+)\s*Inf/s",
    "energy_eff":        r"Energy efficiency:\s+([\d.]+)\s*Inf/s/W",
    "analog_energy_mJ":  r"Analog energy:\s+([\d.]+)\s*mJ",
    "digital_energy_mJ": r"Digital Accelerators energy:\s+([\d.]+)\s*mJ",
    "total_energy_mJ":   r"Total energy:\s+([\d.]+)\s*mJ",
    "avg_power_W":       r"Total avg power:\s+([\d.]+)\s*W",
}

# label -> (parall_type, pipe_type). Labels match the HARP paper's tables.
PIPE_CONFIGS = [
    ("Serial",     "none",        "none"),
    ("IL-Par",     "intra_layer", "none"),
    ("IL-Pipe",    "intra_layer", "intra_layer"),
    ("XL-Pipe",    "intra_layer", "inter_layer"),
    ("IL+XL-Pipe", "intra_layer", "inter_intra_layer"),
    ("XB-pipe",    "intra_layer", "inter_block"),
]

# suite -> arch_model (TABLE III row via set_model_arch.py), CLI args, references.
MODEL_SUITES = {
    "MobileBERT": {
        "arch_model": "MobileBERT",
        "common_args": ["--model", "MobileBERT", "--autoregressive", "false",
                        "--sl", "128", "--chunk_size_d", "8"],
        "expected_latency": {
            "Serial": 25.507772, "IL-Par": 18.735723, "IL-Pipe": 18.242972,
            "XL-Pipe": 7.162998, "IL+XL-Pipe": 4.931997, "XB-pipe": 3.861117,
        },
    },
    "BERT_B": {
        "arch_model": "BERT_B",
        "common_args": ["--model", "BERT_B", "--autoregressive", "false",
                        "--sl", "128", "--chunk_size_d", "8"],
        "expected_latency": {
            "Serial": 31.816359, "IL-Par": 6.436942, "IL-Pipe": 6.1219,
            "XL-Pipe": 5.632124, "IL+XL-Pipe": 3.395197, "XB-pipe": 2.745757,
        },
    },
    "BERT_L": {
        "arch_model": "BERT_L",
        "common_args": ["--model", "BERT_L", "--autoregressive", "false",
                        "--sl", "128", "--chunk_size_d", "8"],
        "expected_latency": {
            "Serial": 80.11219, "IL-Par": 17.337014, "IL-Pipe": 15.904393,
            "XL-Pipe": 19.262568, "IL+XL-Pipe": 10.25465, "XB-pipe": 8.89673,
        },
    },
    "GPT2_small": {
        "arch_model": "GPT2_small",
        "common_args": ["--model", "GPT2_small", "--autoregressive", "false",
                        "--sl", "128", "--chunk_size_d", "8"],
        "expected_latency": {
            "Serial": 31.796708, "IL-Par": 6.417291, "IL-Pipe": 6.102249,
            "XL-Pipe": 5.540376, "IL+XL-Pipe": 3.331856, "XB-pipe": 2.680304,
        },
    },
    "GPT2_small_AR": {
        "arch_model": "GPT2_small",
        # Autoregressive decode: SL=1, chunk_size_d=1, 512-token prefill.
        "common_args": ["--model", "GPT2_small", "--autoregressive", "true",
                        "--sl", "1", "--chunk_size_d", "1", "--prefill_size", "512"],
        "expected_latency": {
            "Serial": 12.933456, "IL-Par": 2.988577, "IL-Pipe": 2.907747,
            "XL-Pipe": 2.988553, "IL+XL-Pipe": 2.907723, "XB-pipe": 2.041827,
        },
    },
    "T5_encoder": {
        "arch_model": "T5_encoder",
        "common_args": ["--model", "T5_encoder", "--autoregressive", "false",
                        "--sl", "128", "--chunk_size_d", "8"],
        # Verified against the HARP paper's T5 Small (chunk 8) table.
        "expected_latency": {
            "Serial": 10.256408, "IL-Par": 2.829653, "IL-Pipe": 2.672132,
            "XL-Pipe": 2.648597, "IL+XL-Pipe": 1.529750, "XB-pipe": 1.238710,
        },
    },
    "T5_decoder": {
        "arch_model": "T5_decoder",
        # Autoregressive decode: SL=1, chunk_size_d=1.
        #
        # The prefill is the encoder context, carried by the hardcoded
        # SL_cross=512 in main.py -- NOT by --prefill_size. --prefill_size must
        # stay <= 1 here: analog_scheduler.py gives the cross-attention K/V
        # projections (fc6/fc7) their real cost only on the `prefill <= 1`
        # branch, and zeroes them out when prefill > 1 (the amortized-KV-cache
        # assumption). 
        "common_args": ["--model", "T5_decoder", "--autoregressive", "true",
                        "--sl", "1", "--chunk_size_d", "1",
                        "--prefill_size", "0"],
        # Verified against the HARP paper's T5 Small decoder table: latency,
        # analog/digital/total energy, avg power, throughput and energy
        # efficiency all match exactly.
        "expected_latency": {
            "Serial": 5.490589, "IL-Par": 2.387793, "IL-Pipe": 2.347378,
            "XL-Pipe": 2.387793, "IL+XL-Pipe": 2.347378, "XB-pipe": 1.421760,
        },
    },
}


def apply_model_arch(arch_model):
    """Rewrite nodes/accelerator_config.py for arch_model before running a suite."""
    proc = subprocess.run([PYTHON, SET_MODEL_ARCH_PATH, "--model", arch_model],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[ARCH] {_color('FAILED', Fore.RED)} to apply arch config for {arch_model}:")
        print(proc.stderr)
        sys.exit(1)
    for line in proc.stdout.strip().splitlines():
        print(f"[ARCH] {line}")


def run_config(common_args, parall_type, pipe_type):
    cmd = [PYTHON, "main.py"] + common_args + [
        "--parall_type", parall_type,
        "--pipe_type", pipe_type,
        "--display_plots", "false",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    return cmd, proc


def parse_metrics(stdout):
    values = {}
    for key, regex in PATTERNS.items():
        m = re.search(regex, stdout)
        values[key] = float(m.group(1)) if m else None
    return values


def pct_diff(actual, expected):
    if expected == 0:
        return 0.0 if actual == 0 else float("inf")
    return abs(actual - expected) / abs(expected) * 100.0


def do_list():
    hdr = (f"{'Suite':<14} {'TABLE III row':<14} {'#ACIM':>6} {'#PMCA+MACE':>11} {'#PMCA':>6} "
           f"{'SRAM(MiB)':>10} {'refs':>5}")
    print(hdr)
    print("-" * len(hdr))
    for name, s in MODEL_SUITES.items():
        n_ref = sum(1 for v in s["expected_latency"].values() if v is not None)
        refs = f"{n_ref}/{len(PIPE_CONFIGS)}"
        row = MODEL_ARCH[resolve_arch_model(s["arch_model"])]
        print(f"{name:<14} {s['arch_model']:<14} {row['acim_tiles']:>6} {row['pmca_red']:>11} "
              f"{row['pmca']:>6} {row['sram_mib']:>10} {refs:>5}")
    print("\nEach ACIM has 8 tiers. See regression_checks/set_model_arch.py --list for the full "
          "TABLE III (incl. link/DDR bandwidth).")
    print("\nargs per suite:")
    for name, s in MODEL_SUITES.items():
        print(f"  {name:<14} {' '.join(s['common_args'])}")
    print(f"\nPipeline configs: {', '.join(l for l, _, _ in PIPE_CONFIGS)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tol", type=float, default=0.5,
                        help="Relative latency tolerance in percent (default: 0.5)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated subset of model suites to run (default: all). "
                             f"Choices: {', '.join(MODEL_SUITES)}")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated subset of pipeline configs to run (default: all). "
                             f"Choices: {', '.join(l for l, _, _ in PIPE_CONFIGS)}")
    parser.add_argument("--list", action="store_true",
                        help="List available model suites and exit.")
    parser.add_argument("--record", action="store_true",
                        help="Print measured latencies as a paste-able expected_latency dict.")
    parser.add_argument("--keep-arch", action="store_true",
                        help="Do not restore nodes/accelerator_config.py on exit.")
    args = parser.parse_args()

    if args.list:
        do_list()
        return

    suite_names = list(MODEL_SUITES)
    if args.models:
        requested = [m.strip() for m in args.models.split(",")]
        unknown = [m for m in requested if m not in MODEL_SUITES]
        if unknown:
            print(f"Unknown model suite(s): {unknown}. Choices: {list(MODEL_SUITES)}")
            sys.exit(2)
        suite_names = requested

    pipe_configs = PIPE_CONFIGS
    if args.configs:
        requested = [c.strip() for c in args.configs.split(",")]
        known = {l for l, _, _ in PIPE_CONFIGS}
        unknown = [c for c in requested if c not in known]
        if unknown:
            print(f"Unknown config(s): {unknown}. Choices: {sorted(known)}")
            sys.exit(2)
        pipe_configs = [c for c in PIPE_CONFIGS if c[0] in requested]

    # Preserve the user's accelerator_config.py -- this script rewrites it.
    # The backup name is PID-unique so concurrent runs cannot clobber each
    # other's copy, and the original bytes are kept in memory as a fallback.
    backup = None
    original_bytes = None
    if not args.keep_arch:
        backup = f"{CONFIG_PATH}.regression_backup.{os.getpid()}"
        with open(CONFIG_PATH, "rb") as f:
            original_bytes = f.read()
        shutil.copy2(CONFIG_PATH, backup)
        print(f"[ARCH] saved current config to {os.path.basename(backup)} "
              f"(restored on exit; --keep-arch to disable)")

    all_results = []
    recorded = {}
    any_fail = False
    total_configs = len(suite_names) * len(pipe_configs)
    done_configs = 0
    start_time = time.time()

    def _write_status(line):
        try:
            with open(STATUS_FILE, "w") as f:
                f.write(line + "\n")
        except OSError:
            pass

    _write_status(f"0/{total_configs} (0.0%) -- starting")

    try:
        for suite_name in suite_names:
            suite = MODEL_SUITES[suite_name]
            expected_latency = suite["expected_latency"]
            recorded[suite_name] = {}

            apply_model_arch(suite["arch_model"])

            for label, parall_type, pipe_type in pipe_configs:
                exp_lat = expected_latency.get(label)
                pct = 100.0 * done_configs / total_configs
                elapsed = time.time() - start_time
                print(f"[RUN {done_configs + 1}/{total_configs} = {pct:.1f}%, elapsed {elapsed:.0f}s] "
                      f"{suite_name:14s} {label:12s} "
                      f"(--parall_type {parall_type} --pipe_type {pipe_type}) ...", flush=True)
                _write_status(f"{done_configs + 1}/{total_configs} ({pct:.1f}%) -- running "
                              f"{suite_name}/{label}, elapsed {elapsed:.0f}s")
                cmd, proc = run_config(suite["common_args"], parall_type, pipe_type)
                done_configs += 1

                if proc.returncode != 0:
                    any_fail = True
                    all_results.append((suite_name, label, "ERROR", None, exp_lat,
                                        f"non-zero exit ({proc.returncode})", cmd))
                    print(f"  -> {_status_text('ERROR')}: process exited {proc.returncode}")
                    tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
                    if tail:
                        print(f"  stderr tail:\n{tail}")
                    continue

                metrics = parse_metrics(proc.stdout)
                actual_lat = metrics.get("latency_ms")
                recorded[suite_name][label] = actual_lat

                if exp_lat is None:
                    status, detail = "PENDING", "no reference value"
                elif actual_lat is None:
                    status, detail = "FAIL", "latency_ms not found in output"
                    any_fail = True
                else:
                    diff = pct_diff(actual_lat, exp_lat)
                    if diff > args.tol:
                        status = "FAIL"
                        any_fail = True
                        detail = f"expected {exp_lat}, got {actual_lat} ({diff:.3f}% diff)"
                    else:
                        status, detail = "PASS", ""

                all_results.append((suite_name, label, status, metrics, exp_lat, detail, cmd))

                if status == "PASS":
                    print(f"  -> {_status_text('PASS')} (latency={actual_lat} ms)")
                elif status == "PENDING":
                    print(f"  -> {_status_text('PENDING')} (actual latency={actual_lat} ms, no reference yet)")
                else:
                    print(f"  -> {_status_text(status)}: {detail}")

                if metrics:
                    print(f"     throughput={metrics.get('throughput')} Inf/s, "
                          f"energy_eff={metrics.get('energy_eff')} Inf/s/W, "
                          f"analog_E={metrics.get('analog_energy_mJ')} mJ, "
                          f"digital_E={metrics.get('digital_energy_mJ')} mJ, "
                          f"total_E={metrics.get('total_energy_mJ')} mJ, "
                          f"avg_power={metrics.get('avg_power_W')} W")
    finally:
        # Restoring the user's config must never itself raise -- a failure here
        # would leave accelerator_config.py holding the last-applied architecture.
        if backup or original_bytes is not None:
            rel = os.path.relpath(CONFIG_PATH, REPO_ROOT)
            try:
                if backup and os.path.exists(backup):
                    shutil.copy2(backup, CONFIG_PATH)
                    os.remove(backup)
                elif original_bytes is not None:
                    # Backup vanished (concurrent run, cleanup, ...): fall back
                    # to the bytes read at startup.
                    with open(CONFIG_PATH, "wb") as f:
                        f.write(original_bytes)
                    print(f"[ARCH] backup file missing; restored {rel} from memory")
                print(f"\n[ARCH] restored original {rel}")
            except OSError as exc:
                print(f"\n[ARCH] WARNING: could not restore {rel}: {exc}", file=sys.stderr)
                print(f"[ARCH] it currently holds the last-applied architecture. "
                      f"Re-apply your own with set_model_arch.py.", file=sys.stderr)

    total_elapsed = time.time() - start_time
    _write_status(f"{total_configs}/{total_configs} (100.0%) -- done, total elapsed {total_elapsed:.0f}s")

    print("\n" + "=" * 100)
    print(f"{'Model':<14} {'Config':<12} {'Status':<8} {'Latency(ms)':>12} {'Throughput':>12} "
          f"{'EnergyEff':>10} {'TotalE(mJ)':>11}")
    for suite_name, label, status, metrics, exp_lat, detail, _ in all_results:
        status_col = _status_text(status, width=8)
        if metrics:
            print(f"{suite_name:<14} {label:<12} {status_col} "
                  f"{str(metrics.get('latency_ms')):>12} {str(metrics.get('throughput')):>12} "
                  f"{str(metrics.get('energy_eff')):>10} {str(metrics.get('total_energy_mJ')):>11}")
        else:
            print(f"{suite_name:<14} {label:<12} {status_col} {'-':>12} {'-':>12} {'-':>10} {'-':>11}  ({detail})")
    print("=" * 100)

    if args.record:
        print("\nMeasured values (paste into MODEL_SUITES[...]['expected_latency']):")
        for suite_name, vals in recorded.items():
            print(f'    "{suite_name}": {{')
            for label, v in vals.items():
                print(f'        "{label}": {v},')
            print("    },")

    if any_fail:
        print(f"\n{_color('MISMATCHES / ERRORS DETECTED:', Fore.RED)}")
        for suite_name, label, status, _, exp_lat, detail, cmd in all_results:
            if status not in ("PASS", "PENDING"):
                print(f"  - {suite_name} / {label}: {detail}")
                print(f"    cmd: {' '.join(cmd)}")
        sys.exit(1)
    else:
        pending = [r for r in all_results if r[2] == "PENDING"]
        print("\n" + _color("All configs with a reference value match within tolerance.", Fore.GREEN)
              + (f" ({len(pending)} PENDING reference data.)" if pending else ""))
        sys.exit(0)


if __name__ == "__main__":
    main()
