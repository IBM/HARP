#!/usr/bin/env python3
#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""
clean_outputs.py

Removes the run artifacts that main.py generates, leaving the directory
structure in place so the next run does not have to recreate it.

By default only `outputs/` is cleaned. Plots are opt-in, since they are usually
the thing you want to keep.

Usage (run from the repo root)
-----
$ python utils/clean_outputs.py                  # clear outputs/
$ python utils/clean_outputs.py --plots          # clear Plots/ as well
$ python utils/clean_outputs.py --all            # same as --plots
$ python utils/clean_outputs.py --dry-run        # show what would go, delete nothing
$ python utils/clean_outputs.py --older-than 7   # only files untouched for 7+ days
$ python utils/clean_outputs.py --model BERT_L   # only files whose name mentions BERT_L
"""

import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Roots holding generated artifacts. Their subdirectories are discovered at run
# time rather than hardcoded: main.py's os.makedirs calls do not cover every
# directory that actually gets written (e.g. outputs/DigitalLayerMapping,
# outputs/HeterogeneousMapping), so a fixed list would silently leave files behind.
OUTPUT_ROOT = "outputs"
PLOTS_ROOT = "Plots"


def discover(root):
    """Return the relative directories to clean under `root`, root itself first.

    Including the root catches stray files written directly into outputs/ or
    Plots/ rather than into a subdirectory.
    """
    abs_root = os.path.join(REPO_ROOT, root)
    if not os.path.isdir(abs_root):
        return []
    subdirs = sorted(
        os.path.join(root, name)
        for name in os.listdir(abs_root)
        if os.path.isdir(os.path.join(abs_root, name))
    )
    return [root] + subdirs


def human(n_bytes):
    """Format a byte count the way du -h would."""
    size = float(n_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024 or unit == "T":
            return f"{size:.0f}{unit}" if unit in ("B", "K") else f"{size:.1f}{unit}"
        size /= 1024


def collect(rel_dir, older_than_days=None, model=None):
    """Return (files, total_bytes) for files directly inside rel_dir.

    Non-recursive: each directory is reported separately by the caller, and the
    directories themselves are always preserved.
    """
    abs_dir = os.path.join(REPO_ROOT, rel_dir)
    if not os.path.isdir(abs_dir):
        return [], 0

    cutoff = None
    if older_than_days is not None:
        cutoff = time.time() - older_than_days * 86400

    files, total = [], 0
    for name in os.listdir(abs_dir):
        path = os.path.join(abs_dir, name)
        if not os.path.isfile(path):
            continue
        if model and model.lower() not in name.lower():
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        if cutoff is not None and st.st_mtime > cutoff:
            continue
        files.append(path)
        total += st.st_size
    return files, total


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plots", action="store_true",
                        help="Also clean Plots/ (kept by default).")
    parser.add_argument("--all", action="store_true",
                        help="Clean both outputs/ and Plots/ (same as --plots).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be removed without deleting anything.")
    parser.add_argument("--older-than", type=float, metavar="DAYS", default=None,
                        help="Only remove files not modified for at least DAYS days.")
    parser.add_argument("--model", type=str, default=None,
                        help="Only remove files whose name contains this model name.")
    args = parser.parse_args()

    targets = discover(OUTPUT_ROOT)
    if args.plots or args.all:
        targets += discover(PLOTS_ROOT)
    if not targets:
        print(f"  Nothing to clean: {OUTPUT_ROOT}/ does not exist.")
        return

    grand_files, grand_bytes, removed, failed = 0, 0, 0, []
    rows = []

    for rel_dir in targets:
        files, nbytes = collect(rel_dir, args.older_than, args.model)
        rows.append((rel_dir, len(files), nbytes))
        grand_files += len(files)
        grand_bytes += nbytes

        if not args.dry_run:
            for path in files:
                try:
                    os.remove(path)
                    removed += 1
                except OSError as exc:
                    failed.append((path, exc))

    width = max((len(r[0]) for r in rows), default=20)
    for rel_dir, count, nbytes in rows:
        note = "" if os.path.isdir(os.path.join(REPO_ROOT, rel_dir)) else "  (missing)"
        print(f"  {rel_dir:<{width}}  {count:>5} files  {human(nbytes):>8}{note}")
    print("  " + "-" * (width + 24))

    filters = []
    if args.older_than is not None:
        filters.append(f"older than {args.older_than:g}d")
    if args.model:
        filters.append(f"model~{args.model}")
    suffix = f"  [{', '.join(filters)}]" if filters else ""

    if args.dry_run:
        print(f"  dry run: would remove {grand_files} files, {human(grand_bytes)}{suffix}")
    else:
        print(f"  Removed {removed} files, reclaimed {human(grand_bytes)}{suffix}")

    if failed:
        print(f"\n  {len(failed)} file(s) could not be removed:", file=sys.stderr)
        for path, exc in failed[:10]:
            print(f"    {os.path.relpath(path, REPO_ROOT)}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not (args.plots or args.all):
        kept, kept_bytes = 0, 0
        for rel_dir in discover(PLOTS_ROOT):
            f, b = collect(rel_dir)
            kept += len(f)
            kept_bytes += b
        if kept:
            print(f"  Plots/ left untouched ({kept} files, {human(kept_bytes)}) -- use --plots to clean it too.")


if __name__ == "__main__":
    main()
