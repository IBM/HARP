#!/usr/bin/env python3
#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#
import argparse


def none_or_int(x):
    """
    Convert a value to int or None if the string is 'none'.
    """
    if isinstance(x, str) and x.lower() == 'none':
        return None
    return int(x)

def str2bool(v):
    """
    Convert string to boolean. Accepts: 'true','t','1' or 'false','f','0'.
    """
    if isinstance(v, bool):
        return v
    low = v.lower()
    if low in ('true', '1', 't'):
        return True
    if low in ('false', '0', 'f'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Configure model parameters via CLI arguments'
    )

    parser.add_argument(
        '--model',
        choices=['MobileBERT', 'BERT_B', 'BERT_L', 'NanoGPT', 'GPT2_small', 'dumpLayer', 'dumpModel', 'T5_encoder', 'T5_decoder', 'ALBERT_B'],
        default='MobileBERT',
        help='Model to use (default: MobileBERT)'
    )

    parser.add_argument(
        '--dumplayer_size',
        choices=[64, 128, 512, 1024, 2048, 4096, 8192],
        type=int,
        default=128,
        help='Layer size for dumpLayer'
    )
    
    parser.add_argument(
        '--sl',
        type=int,
        default=128,
        help='Sequence Length (default: 128)'
    )

    parser.add_argument(
        '--autoregressive',
        type=str2bool,
        choices=[True, False],
        default=False,
        help='Enable autoregressive generation for Decoders (default: False)'
    )

    parser.add_argument(
        '--prefill_size',
        type=int,
        default=512,
        help='Prefill size for autoregressive models (default: 512)'
    )

    parser.add_argument(
        '--opt_level',
        choices=['latency', 'area', 'power', 'area_latency_balance'],
        default='latency',
        help='Optimization level (default: latency)'
    )

    parser.add_argument(
        '--parall_type',
        choices=['none', 'intra_layer'],
        default='none',
        help='Parallelization type (default: none)'
    )

    parser.add_argument(
        '--pipe_type',
        choices=[
            'none', 'intra_layer', 'inter_layer',
            'inter_intra_layer', 'inter_block'
        ],
        default='none',
        help='Pipelining type (default: none)'
    )

    parser.add_argument(
        '--flash_attention',
        type=str2bool,
        choices=[True, False],
        default=False,
        help='Enable flash attention (default: False)'
    )

    parser.add_argument(
        '--sl_warp',
        type=none_or_int,
        default=none_or_int(str(int(128/2))),
        help='SL warp size (int) or "none" (default: 64)'
    )

    parser.add_argument(
        '--auto_tune_sl_warp',
        type=str2bool,
        choices=[True, False],
        default=False,
        help='Enable auto-tuning of SL warp (default: False)'
    )

    parser.add_argument(
        '--chunk_opt',
        type=str2bool,
        choices=[True, False],
        default=False,
        help='Enable chunk optimization (default: False)'
    )

    parser.add_argument(
        '--chunk_size_d',
        type=int,
        default=8,
        help='Chunk size D (default: 8)'
    )


    parser.add_argument(
        '--use_transfer_link',
        type=str2bool,
        choices=[True, False],
        default=True,
        help='Enable or disable transfer link (default: True)'
    )

    parser.add_argument(
        '--display_plots',
        type=str2bool,
        choices=[True, False],
        default=False,
        help='Enable plot display (default: False)'
    )

    parser.add_argument(
        '--digital_opt',
        choices=['latency', 'power'],
        default='latency',
        help='Optimization level (default: latency)'
    )

    return parser.parse_args()
