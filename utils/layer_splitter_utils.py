#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#

from nodes.accelerator_config import *
import math

SUFFIX = "_TILE_CONFIG"
EXCLUDE_BASES = {"ACIM", "DDR", "SRAM", "LINK"}  # exclude these families

def _base_from_cfg_name(cfg_name: str) -> str:
    return cfg_name[:-len(SUFFIX)]  # e.g., "PMCA_TILE_CONFIG" -> "PMCA"

def discover_accelerators(cfg_globals=None):
    """
    Return list of {'base': <family/type_tag>, 'cfg': <dict>} for all compute families
    discovered from *_TILE_CONFIG in accelerator_config, excluding ACIM/DDR/DRAM/SRAM/LINK.
    """
    if cfg_globals is None:
        cfg_globals = globals()
    families = []
    for name, cfg in cfg_globals.items():
        if isinstance(name, str) and name.endswith(SUFFIX) and isinstance(cfg, dict):
            base = _base_from_cfg_name(name)
            if base in EXCLUDE_BASES:
                continue
            families.append({"base": base, "cfg": cfg})
    return families

def acc_has_op(base: str, op_name: str, accelerators) -> bool:
    for acc in accelerators:
        if getattr(acc, "type_tag", "") == base:
            ops = getattr(acc, "supported_ops", []) or []
            if op_name in ops:
                return True
    return False

def effective_cores_for_op(cfg: dict, op_name: str) -> int:
    per_op = cfg.get("num_cores_by_op", {})
    if op_name in per_op: return int(per_op[op_name])
    if op_name == "layernorm": return int(cfg.get("num_cores_row", cfg.get("num_cores", 1)))
    if op_name == "gelu":      return int(cfg.get("num_cores_col", cfg.get("num_cores", 1)))
    if op_name == "atten":     return int(cfg.get("max_size_vect_parall", cfg.get("num_cores", 1)))
    return int(cfg.get("num_cores", 1))

def max_cores_for_op(op_name: str, accelerators, cfg_globals=None, default=1):
    cores = []
    for fam in discover_accelerators(cfg_globals):
        cfg, base = fam["cfg"], fam["base"]
        if cfg.get("num_tiles", 0) <= 0: continue
        if not acc_has_op(base, op_name, accelerators): continue
        cores.append(effective_cores_for_op(cfg, op_name))
    return max(cores) if cores else default

def get_min_tcdm_capacity_for_op(op_name: str, accelerators, cfg_globals=None, default=0):
    cands = []
    for fam in discover_accelerators(cfg_globals):
        cfg, base = fam["cfg"], fam["base"]
        if cfg.get("num_tiles", 0) <= 0: continue
        if not acc_has_op(base, op_name, accelerators): continue
        if "tcdm_capacity" in cfg: cands.append(cfg["tcdm_capacity"])
    return min(cands) if cands else default

# =============================== #
#    Internal Split Utilities     #
# =============================== #
def split_layer_intra(layer, accelerators_instances, digital_opt):
    """
    Split a layer (LayerNorm or GeLU) for intra-layer parallelism.

    Args:
        layer: The layer to split.
        num_pmcas: Maximum number of digital PMCAs available.

    Returns:
        A list of split sublayers.
    """
    if layer.type == "atten":
        return split_attention_heads(layer, accelerators_instances, digital_opt)

    elif layer.type in ("layernorm", "gelu", "gemm"):
        sl, hidden_size = layer.input_shape
        _, out_shape = layer.output_shape

        if layer.type == "layernorm":
            min_rows = max_cores_for_op("layernorm", accelerators_instances)
            ideal_splits = sl // min_rows
        elif layer.type == "gelu":
            min_cols = max_cores_for_op("gelu", accelerators_instances)
            ideal_splits = hidden_size // min_cols
        elif layer.type == "gemm":
            min_rows = max_cores_for_op("layernorm", accelerators_instances)
            ideal_splits = sl // min_rows ##we split across N dimension otherwise they do not fit 
        else:
            raise ValueError(f"Unsupported layer type for smart split: {layer.type}")

        # Now clip the number of splits to available PMCAs that supports that operation
        acc_coll ={}
        for i in accelerators_instances:
            if i.type_tag in acc_coll or i.type_tag in ("Link", "DDR"):
                continue
            if layer.type in i.supported_ops:
                acc_coll[i.type_tag] = [i.estimate_cost(layer),i.estimate_power(layer)]

        if acc_coll:  # at least one accelerator supported this layer
            if digital_opt == "latency":
                # pick lowest latency, tie-break on power
                selected_tag = min(acc_coll, key=lambda t: (acc_coll[t][0], acc_coll[t][1]))
            else:  # digital_opt == "power"
                # pick lowest power, tie-break on latency
                selected_tag = min(acc_coll, key=lambda t: (acc_coll[t][1], acc_coll[t][0]))   
            num_selected_type = sum(1 for acc in accelerators_instances if acc.type_tag == selected_tag)
        else:
            raise RuntimeError(f"No accelerator supports layer type {layer.type}")
        num_splits = min(num_selected_type, max(1, ideal_splits))

        # If even one split would create very tiny ops, force no split
        min_split_size = min_rows if layer.type in ("layernorm", "gemm") else min_cols
        if (layer.type == "layernorm" and sl // num_splits < min_split_size) or \
        (layer.type == "gelu" and hidden_size // num_splits < min_split_size) or \
        (layer.type == "gemm" and sl // num_splits < min_split_size):
            num_splits = 1

        sublayers = []
        if layer.type == "layernorm":
            rows_per_split = math.ceil(sl / num_splits)
            cols_per_split = hidden_size
        elif layer.type == "gemm":
            rows_per_split = math.ceil(sl / num_splits)
            cols_per_split = hidden_size
        else:
            rows_per_split = sl
            cols_per_split = math.ceil(hidden_size / num_splits)


        for i in range(num_splits):
            if layer.type == "layernorm":
                sub_input_shape = (rows_per_split, hidden_size)
                sub_output_shape = (rows_per_split, hidden_size)
            elif layer.type == "gemm":
                sub_input_shape = (rows_per_split, hidden_size)
                sub_output_shape = (rows_per_split, out_shape)
            else:  # gelu
                sub_input_shape = (sl, cols_per_split)
                sub_output_shape = (sl, cols_per_split)

            sublayer = type(layer)(
                name=f"{layer.name}_split{i}",
                input_shape=sub_input_shape,
                output_shape=sub_output_shape,
                layer_type=layer.type,
                attention_type= layer.attention_type
            )
            sublayers.append(sublayer)

        return sublayers

    else:
        raise ValueError(f"Cannot split unknown layer type: {layer.type}")


from dataclasses import dataclass
from typing import Tuple

@dataclass
class DummyLayer:
    type: str
    input_shape: Tuple[int, int]
    name: str = ""
    output_shape: Tuple[int, int] = None


def split_attention_heads(layer, accelerators_instances, digital_opt):
    """
    Split an attention layer by distributing its heads across multiple PMCAs.

    Args:
        layer: The attention layer to split.
        max_pmcas: The maximum number of PMCA_RED instances available for splitting.

    Returns:
        A list of sublayers, each containing a subset of the attention heads.
    """
    sl, head_dim = layer.input_shape
    _, hidden_size = layer.output_shape
    num_heads = hidden_size // head_dim

    softmax_layer = DummyLayer(
    type="softmax",
    input_shape=(sl, sl)
    )
    gemm_layer = DummyLayer(
    type="gemm",
    input_shape=(sl, sl, head_dim)
    )

    acc_coll ={}
    for i in accelerators_instances:
        if i.type_tag in acc_coll or i.type_tag in ("Link", "DDR"):
            continue
        if layer.type in i.supported_ops:
            acc_coll[i.type_tag] = [i.estimate_cost(softmax_layer)+i.estimate_cost(gemm_layer),i.estimate_power(softmax_layer)+i.estimate_power(gemm_layer)]

    if acc_coll:  # at least one accelerator supported this layer
        if digital_opt == "latency":
            # pick lowest latency, tie-break on power
            selected_tag = min(acc_coll, key=lambda t: (acc_coll[t][0], acc_coll[t][1]))
        else:  # digital_opt == "power"
            # pick lowest power, tie-break on latency
            selected_tag = min(acc_coll, key=lambda t: (acc_coll[t][1], acc_coll[t][0]))   
        num_selected_type = sum(1 for acc in accelerators_instances if acc.type_tag == selected_tag)
    else:
        raise RuntimeError(f"No accelerator supports layer type {layer.type}")

    if selected_tag in ("DA0"):
        num_splits = min(num_selected_type, math.ceil(num_heads/(DA0_TILE_CONFIG["num_cores"]/(head_dim/DA0_TILE_CONFIG["max_parallel"]))))  #DA0 max_parallel limits the number of heads that can be processed in parallel
    else:
        num_splits = min(num_selected_type, num_heads)

    heads_per_split = num_heads // num_splits
    extra_heads = num_heads % num_splits

    sublayers = []
    head_start = 0

    for i in range(num_splits):
        heads_assigned = heads_per_split + (1 if i < extra_heads else 0)
        if heads_assigned == 0:
            continue

        sub_head_dim = head_dim
        sub_hidden_size = heads_assigned * sub_head_dim

        sublayer = type(layer)(
            name=f"{layer.name}_head{head_start}_to_head{head_start + heads_assigned - 1}",
            input_shape=(sl, sub_head_dim),
            output_shape=(sl, sub_hidden_size),
            layer_type=layer.type,
            attention_type= layer.attention_type
        )
        sublayers.append(sublayer)

        head_start += heads_assigned

    return sublayers
