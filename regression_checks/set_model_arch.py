#!/usr/bin/env python3
#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""
set_model_arch.py

Applies the per-model architecture configurations from TABLE III of the HARP
paper to nodes/accelerator_config.py, in place.

    E. Ferro, H. Benmeziane and I. Boybat, "HARP: Heterogeneous Analog-Digital
    Resource-Aware Performance and Scheduling Framework for Transformer
    Acceleration," IEEE TPDS, vol. 37, no. 9, pp. 2107-2121, 2026.
    doi: 10.1109/TPDS.2026.3706975

Run this BEFORE main.py. It only edits the config file on disk; main.py itself
does no per-model switching -- it just reads whatever is currently in
accelerator_config.py, same as editing the file by hand.

SRAM_TILE_CONFIG["size"] stays fixed at 1 MiB/tile, so "SRAM size (MiB)" from
the paper maps directly onto SRAM_TILE_CONFIG["num_tiles"].

Usage
-----
$ python regression_checks/set_model_arch.py --model BERT_B
$ python regression_checks/set_model_arch.py --model T5_encoder
$ python regression_checks/set_model_arch.py --list
$ python regression_checks/set_model_arch.py --model BERT_B --check
$ python regression_checks/set_model_arch.py --check-all
"""

import argparse
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(REPO_ROOT, "nodes", "accelerator_config.py")

# TABLE III -- Architecture Configurations (HARP, IEEE TPDS 2026).
# Values are (dict_name, key) -> value. "Each ACIM has 8 tiers."
MODEL_ARCH = {
    "MobileBERT": {"acim_tiles": 16,  "pmca_red": 4,  "pmca": 0,  "sram_mib": 1,  "link_bw": 512, "ddr_bw": 800},
    "BERT_B":     {"acim_tiles": 96,  "pmca_red": 12, "pmca": 20, "sram_mib": 8,  "link_bw": 512, "ddr_bw": 800},
    "BERT_L":     {"acim_tiles": 144, "pmca_red": 12, "pmca": 20, "sram_mib": 12, "link_bw": 512, "ddr_bw": 800},
    "GPT2_small": {"acim_tiles": 96,  "pmca_red": 12, "pmca": 20, "sram_mib": 8,  "link_bw": 512, "ddr_bw": 800},
    # T5_Small is a single row in TABLE III; it covers both the encoder and the
    # decoder, which share the same platform sizing.
    "T5_small":   {"acim_tiles": 96,  "pmca_red": 12, "pmca": 20, "sram_mib": 8,  "link_bw": 512, "ddr_bw": 800},
}

# main.py --model values that map onto a TABLE III row.
MODEL_ALIASES = {
    "T5_encoder": "T5_small",
    "T5_decoder": "T5_small",
}

# logical field -> (dict name in accelerator_config.py, key within that dict)
FIELD_TO_CONFIG = {
    "acim_tiles": ("ACIM_TILE_CONFIG", "num_tiles"),
    "pmca_red":   ("PMCA_RED_TILE_CONFIG", "num_tiles"),
    "pmca":       ("PMCA_TILE_CONFIG", "num_tiles"),
    "sram_mib":   ("SRAM_TILE_CONFIG", "num_tiles"),
    "link_bw":    ("LINK_CONFIG", "bandwidth_b_per_cycle"),
    "ddr_bw":     ("DDR_TILE_CONFIG", "bandwidth"),
}

# TABLE III footnote: each ACIM has 8 tiers. Applied for every model.
FIXED_FIELDS = {
    ("ACIM_TILE_CONFIG", "num_tiers"): 8,
}

PRETTY = {
    "acim_tiles": "#ACIM tiles",
    "pmca_red":   "#PMCA + MACE nodes",
    "pmca":       "#PMCA nodes",
    "sram_mib":   "SRAM size (MiB)",
    "link_bw":    "Link bandwidth (b/cycles)",
    "ddr_bw":     "DDR bandwidth (MiB/s)",
}


def resolve(model):
    """Map a --model value onto its TABLE III row name."""
    return MODEL_ALIASES.get(model, model)


def _find_key_line(lines, dict_name, key):
    """Return the index of the `"key": <num>` line inside the `dict_name` block."""
    start_pat = re.compile(rf"^{re.escape(dict_name)}\s*=\s*\{{")
    key_pat = re.compile(rf'^(\s*"{re.escape(key)}"\s*:\s*)([\d.]+)(.*)$')

    in_block = False
    for i, line in enumerate(lines):
        if not in_block:
            if start_pat.match(line):
                in_block = True
            continue
        if line.strip() == "}":
            raise RuntimeError(f'Reached end of {dict_name} without finding "{key}"')
        m = key_pat.match(line)
        if m:
            return i, m
    raise RuntimeError(f"Could not find dict block {dict_name!r} in {CONFIG_PATH}")


def read_value(lines, dict_name, key):
    _, m = _find_key_line(lines, dict_name, key)
    raw = m.group(2)
    return int(float(raw)) if float(raw).is_integer() else float(raw)


def set_value(lines, dict_name, key, new_value):
    i, m = _find_key_line(lines, dict_name, key)
    old_value = m.group(2)
    lines[i] = f"{m.group(1)}{new_value}{m.group(3)}\n"
    return old_value


def targets_for(row):
    """Yield (dict_name, key, expected_value, label) for a TABLE III row."""
    for field, value in row.items():
        dict_name, key = FIELD_TO_CONFIG[field]
        yield dict_name, key, value, PRETTY[field]
    for (dict_name, key), value in FIXED_FIELDS.items():
        yield dict_name, key, value, "ACIM tiers"


def do_list():
    hdr = f"{'Model':<12} {'#ACIM':>6} {'#PMCA+MACE':>11} {'#PMCA':>6} {'SRAM(MiB)':>10} {'Link(b/cyc)':>12} {'DDR(MiB/s)':>11}"
    print("TABLE III -- Architecture Configurations (HARP, IEEE TPDS 2026)")
    print(hdr)
    print("-" * len(hdr))
    for name, r in MODEL_ARCH.items():
        print(f"{name:<12} {r['acim_tiles']:>6} {r['pmca_red']:>11} {r['pmca']:>6} "
              f"{r['sram_mib']:>10} {r['link_bw']:>12} {r['ddr_bw']:>11}")
    print("\nEach ACIM has 8 tiers.")
    if MODEL_ALIASES:
        print("Aliases: " + ", ".join(f"{k} -> {v}" for k, v in MODEL_ALIASES.items()))


def do_check(lines, model, quiet=False):
    """Compare the config on disk against a model's TABLE III row.
    Returns True if every field matches."""
    row = MODEL_ARCH[resolve(model)]
    mismatches = []
    for dict_name, key, expected, label in targets_for(row):
        actual = read_value(lines, dict_name, key)
        if actual != expected:
            mismatches.append((label, dict_name, key, actual, expected))

    if not quiet:
        if mismatches:
            print(f"MISMATCH: nodes/accelerator_config.py does not match TABLE III for {model}")
            for label, dict_name, key, actual, expected in mismatches:
                print(f'  {label:<26} {dict_name}["{key}"]: {actual} (expected {expected})')
        else:
            print(f"OK: nodes/accelerator_config.py matches TABLE III for {model}")
    return not mismatches


def do_check_all(lines):
    """Report which TABLE III rows the current config matches, if any."""
    matched = [name for name in MODEL_ARCH if do_check(lines, name, quiet=True)]
    print("Current nodes/accelerator_config.py vs TABLE III:")
    for name in MODEL_ARCH:
        ok = name in matched
        print(f"  {name:<12} {'MATCH' if ok else 'differs'}")
    if matched:
        alias_note = [a for a, t in MODEL_ALIASES.items() if t in matched]
        extra = f" (also: {', '.join(alias_note)})" if alias_note else ""
        print(f"\nConfig currently corresponds to: {', '.join(matched)}{extra}")
    else:
        print("\nConfig does not correspond to any TABLE III row.")
    return bool(matched)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choices = sorted(set(MODEL_ARCH) | set(MODEL_ALIASES))
    parser.add_argument("--model", choices=choices,
                        help="Model whose TABLE III configuration to apply.")
    parser.add_argument("--check", action="store_true",
                        help="Verify the config on disk against --model instead of writing it. "
                             "Exits 1 on mismatch.")
    parser.add_argument("--check-all", action="store_true",
                        help="Report which TABLE III row the current config corresponds to.")
    parser.add_argument("--list", action="store_true",
                        help="Print TABLE III and exit.")
    args = parser.parse_args()

    if args.list:
        do_list()
        return

    if not os.path.exists(CONFIG_PATH):
        print(f"Config not found: {CONFIG_PATH}", file=sys.stderr)
        sys.exit(2)

    with open(CONFIG_PATH) as f:
        lines = f.readlines()

    if args.check_all:
        sys.exit(0 if do_check_all(lines) else 1)

    if not args.model:
        parser.error("one of --model, --list or --check-all is required")

    if args.check:
        sys.exit(0 if do_check(lines, args.model) else 1)

    row = MODEL_ARCH[resolve(args.model)]
    changes = []
    for dict_name, key, value, label in targets_for(row):
        old_value = set_value(lines, dict_name, key, value)
        if str(old_value) != str(value):
            changes.append(f'{dict_name}["{key}"]: {old_value} -> {value}')

    with open(CONFIG_PATH, "w") as f:
        f.writelines(lines)

    rel = os.path.relpath(CONFIG_PATH, REPO_ROOT)
    print(f"Updated {rel} for model={args.model} (TABLE III row: {resolve(args.model)}):")
    if changes:
        for c in changes:
            print(f"  {c}")
    else:
        print("  (already matched TABLE III -- no values changed)")


if __name__ == "__main__":
    main()
