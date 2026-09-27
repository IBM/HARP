#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#

import networkx as nx
from nodes.accelerator_config import *
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import logging
from colorama import Fore, Style, init
from utils.plots import plot_schedule_timeline
import os
import warnings

init(autoreset=True)

def save_schedule_txt(schedule_dict, filename="schedule_summary.txt"):
    """
    Write a simple human-readable dump of schedule_dict to `filename`.
    """
    with open(filename, "w") as f:
        f.write("=== Schedule Summary ===\n\n")
        for acc, tasks in schedule_dict.items():
            f.write(f"Accelerator: {acc}\n")
            f.write("-" * (12 + len(acc)) + "\n")
            for layer, start, dur in tasks:
                f.write(f"  {layer:10s}  start={start:7.9f}  dur={dur:7.9f}\n")
            f.write("\n")
    print(f"Wrote schedule to {filename}")

def summarize_energy(accel_instances, total_lat, analog_pools, breakdown=False):
    """
    Match the behavior of the original summarize_energy (the 'correct' one),
    but optionally return breakdowns by type and by instance.

    Returns:
        If breakdown == False:
            (analog_E_mJ, digital_E_mJ, total_E_mJ, avg_P_W)
        If breakdown == True:
            (analog_E_mJ, digital_E_mJ, total_E_mJ, avg_P_W, breakdown_dict)
    """
    analog_E = 0.0
    digital_E = 0.0

    # Matches original de-dup state and semantics
    list_of_tuple_acc_fcname = []

    # For breakdowns
    energy_by_type = {}
    energy_by_instance = {}

    for acc in accel_instances:
        new_start = -1  # reset per-acc, as in the original
        for layer_name, start, dur in acc.schedule:
            # Skip consecutive duplicate entries for same start (non-Link), as in original
            if start == new_start and acc.type_tag != "Link":
                continue
            
            if acc.type_tag == "ACIMTile":
                ppu_power = analog_pools[layer_name]["estimated_power_ppu"]
                if not layer_name.startswith("residual"):
                    latency_1 = analog_pools[layer_name].get("latency", 1.0)
                    ppu_latency_1 = analog_pools[layer_name].get("ppu_latency", 0.0)
                    extra_base_1 = analog_pools[layer_name].get("extra_base", 0.0)
                    extra_ppu_1 = analog_pools[layer_name].get("extra_ppu", 0.0)

                    full_acim_power = analog_pools[layer_name]["estimated_power"]
                    extra_acim_power = analog_pools[layer_name].get("extra_estimated_power", 0.0)
                    base_acim_power = max(0.0, full_acim_power - extra_acim_power)
                    extra_ppu_power = analog_pools[layer_name].get("extra_ppu_power", 0.0)

                    ppu_frac = ppu_latency_1 / latency_1
                    extra_base_frac = extra_base_1 / latency_1
                    extra_ppu_frac = extra_ppu_1 / latency_1

                    ppu_dur = dur * ppu_frac
                    extra_base_dur = dur * extra_base_frac
                    extra_ppu_dur = dur * extra_ppu_frac
                    extra_tile_dur = max(0.0, extra_base_dur - extra_ppu_dur)
                    base_tile_dur = max(0.0, dur - ppu_dur - extra_tile_dur)
                    base_ppu_dur = max(0.0, ppu_dur - extra_ppu_dur)

                    E_mJ = (base_tile_dur * base_acim_power
                            + extra_tile_dur * extra_acim_power
                            + base_ppu_dur * ppu_power
                            + extra_ppu_dur * extra_ppu_power)
                else:
                    E_mJ = dur * ppu_power                    # ms * W = mJ
            else:
                power_W = acc.power_model(layer_name)  # W
                E_mJ = dur * power_W                   # ms * W = mJ

            # ---- ACIM de-dup block (replicates original logic) ----
            if acc.type_tag == "ACIMTile":
                if any(layer_name in elem for tup in list_of_tuple_acc_fcname for elem in tup) and not (acc.name, layer_name) in list_of_tuple_acc_fcname:
                    # Skip this entry entirely (no energy or breakdown counted),
                    # and update new_start to match original control flow
                    continue
                if (acc.name, layer_name) not in list_of_tuple_acc_fcname:
                    list_of_tuple_acc_fcname.append((acc.name, layer_name))
                analog_E += E_mJ
            else:
                digital_E += E_mJ

            # ---- breakdowns (only for entries we actually counted) ----
            energy_by_type.setdefault(acc.type_tag, 0.0)
            energy_by_type[acc.type_tag] += E_mJ

            if acc.name not in energy_by_instance:
                energy_by_instance[acc.name] = {"type": acc.type_tag, "energy_mJ": 0.0}
            energy_by_instance[acc.name]["energy_mJ"] += E_mJ

            new_start = start

    total_E_mJ = analog_E + digital_E
    avg_P_W = (total_E_mJ / total_lat) if total_lat > 0 else 0.0

    if not breakdown:
        return analog_E, digital_E, total_E_mJ, avg_P_W

    # Build breakdown dict with avg power derived the same way
    power_by_type = {t: (e_mJ / total_lat) if total_lat > 0 else 0.0
                     for t, e_mJ in energy_by_type.items()}
    power_by_instance = {n: (entry["energy_mJ"] / total_lat) if total_lat > 0 else 0.0
                         for n, entry in energy_by_instance.items()}

    breakdown_dict = {
        "by_type": {t: {"energy_mJ": energy_by_type[t], "avg_power_W": power_by_type[t]}
                    for t in energy_by_type},
        "by_instance": {n: {"type": energy_by_instance[n]["type"],
                            "energy_mJ": energy_by_instance[n]["energy_mJ"],
                            "avg_power_W": power_by_instance[n]}
                        for n in energy_by_instance}
    }

    return analog_E, digital_E, total_E_mJ, avg_P_W, breakdown_dict




def summarize_area(analog_tiles, pmca_red_tiles, pmca_tiles, das_tiles, sram_tiles):
    analog_A = analog_tiles * ACIM_TILE_CONFIG["area"]
    digital_A = pmca_red_tiles * PMCA_RED_TILE_CONFIG["area"]+pmca_tiles*PMCA_TILE_CONFIG["area"]+sram_tiles*SRAM_TILE_CONFIG["area"]+das_tiles*DA0_TILE_CONFIG["area"]
    TotalArea = analog_A + digital_A
    return TotalArea

def summarize_required_sram(hidden_size, SL, PREFILL, num_blocks, model_name, autoregressive, SL_CROSS=None):
    if model_name in ['BERT_B', 'BERT_L', 'MobileBERT', "T5_encoder", "ALBERT_B"]: #Encoder only model
        size_attention = hidden_size*SL * 4 * ACIM_TILE_CONFIG["precision"]/8/(1024**2) #size in MiB: only single block attention needs to be stored
    elif model_name in ['NanoGPT', "GPT2_small"]:
        if not autoregressive :
            size_attention = hidden_size*SL * 4 * ACIM_TILE_CONFIG["precision"]/8/(1024**2) #size in MiB: only single block attention needs to be stored
        else:
            size_attention = num_blocks*(hidden_size*SL * 2 + hidden_size*PREFILL*2) * ACIM_TILE_CONFIG["precision"]/8/(1024**2) #size in MiB
    elif model_name in ['T5_decoder']: #T5_decoder 
        # T5 decoder has, per block:
        # 1) self-attention:
        #    Q + out           -> hidden_size * SL * 2
        #    K + V cache       -> hidden_size * PREFILL * 2
        #
        # 2) cross-attention:
        #    Q + out           -> hidden_size * SL * 2
        #    K + V encoder mem -> hidden_size * SL_CROSS * 2
        #
        # Total per block:
        #    hidden_size * (4*SL + 2*PREFILL + 2*SL_CROSS)
        size_attention = num_blocks * (
            hidden_size * SL * 4 +
            hidden_size * PREFILL * 2 +
            hidden_size * SL_CROSS * 2
        ) *  ACIM_TILE_CONFIG["precision"]/8/(1024**2) #size in MiB
    else :
        warnings.warn(f"No attention layer in the selected model / Unknown model name {model_name} for SRAM estimation.")
        size_attention = 0
    num_sram = math.ceil(size_attention/SRAM_TILE_CONFIG["size"])
    return num_sram, size_attention

def summarize_TDPsystem(num_tiles_used):
    analog_P = (ACIM_TILE_CONFIG['power']+ACIM_TILE_CONFIG['power_ppu'])*num_tiles_used #* ACIM_TILE_CONFIG['num_tiers']
    digital_P = PMCA_RED_TILE_CONFIG['num_tiles']*PMCA_RED_TILE_CONFIG['power'] + PMCA_TILE_CONFIG['num_tiles']*PMCA_TILE_CONFIG['power']+SRAM_TILE_CONFIG['num_tiles']*SRAM_TILE_CONFIG['power']+LINK_CONFIG['num_links']*LINK_CONFIG['power']
    TDP = analog_P + digital_P
    return TDP #W


def summarize_Areasystem():
    analog_A = ACIM_TILE_CONFIG['area']*(ACIM_TILE_CONFIG['num_tiles'])
    digital_A = PMCA_RED_TILE_CONFIG['num_tiles']*PMCA_RED_TILE_CONFIG['area'] + PMCA_TILE_CONFIG['num_tiles']*PMCA_TILE_CONFIG['area']+SRAM_TILE_CONFIG['num_tiles']*SRAM_TILE_CONFIG['area']
    TotalArea = analog_A + digital_A
    return TotalArea #mm2

def detect_share_inputs(graph):
    """
    Returns a dict layer_name -> set(of other layer_names)
    for all nodes that share at least one common predecessor.
    Also includes nodes with zero predecessors (preds == ()),
    which are considered to share the same "input set".
    """
    pred_to_nodes = defaultdict(set)   # pred -> set(nodes using pred)
    no_pred_nodes = []                 # nodes with preds == ()

    for node in graph.nodes:
        preds = list(graph.predecessors(node))
        if not preds:
            no_pred_nodes.append(node)
        else:
            for p in preds:
                pred_to_nodes[p].add(node)

    share_map = defaultdict(set)       # node -> set(peers)
    #1 ) Share if they share at least one predecessor
    for nodes in pred_to_nodes.values():
        if len(nodes) > 1:
            for n in nodes:
                share_map[n] |= (nodes - {n})

    # 2) Preserve old behavior for nodes with no predecessors (e.g., emb_word, emb_pos)
    if len(no_pred_nodes) > 1:
        no_pred_set = set(no_pred_nodes)
        for n in no_pred_nodes:
            share_map[n] |= (no_pred_set - {n})
    
    return dict(share_map)

def _merged_time_len(intervals):
    """Return total covered time by the union of [start, end) intervals."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals, key=lambda x: x[0])
    total = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:            # overlap/contiguous
            cur_e = max(cur_e, e)
        else:                     # disjoint
            total += (cur_e - cur_s)
            cur_s, cur_e = s, e
    total += (cur_e - cur_s)
    return total

def schedule_summary_and_plots(schedule_dict, accelerators, scheduler, model_name,
                               SL, prefill, total_latency, mapping_summary, num_blocks, sl_full, 
                               hidden_size, max_tile_used_single_tier, max_tile_used, display_plots=True, autoregressive=False, SL_CROSS=None):
    used_pmca_red = {acc.name for acc in accelerators if acc.type_tag == "PMCA_RED" and acc.schedule}
    used_pmca     = {acc.name for acc in accelerators if acc.type_tag == "PMCA" and acc.schedule}
    used_da0     = {acc.name for acc in accelerators if acc.type_tag == "DA0" and acc.schedule}

    # Plot
    if display_plots:
        plot_schedule_timeline(schedule_dict, model_name, SL, scheduler.optimization_level, scheduler.pipelining_type, scheduler.parallelism_type, scheduler.flash_attention)

    system_TDP = summarize_TDPsystem(ACIM_TILE_CONFIG['num_tiles'])
    system_Area = summarize_Areasystem()

    required_sram_tiles, sram_usage = summarize_required_sram(hidden_size, sl_full, prefill, num_blocks, model_name, autoregressive, SL_CROSS)
    ae, de, te, power, breakdown_dict = summarize_energy(accelerators, total_latency, mapping_summary, breakdown=True)
    area_usage = summarize_area(max_tile_used_single_tier,len(used_pmca_red),len(used_pmca),len(used_da0), min(required_sram_tiles,SRAM_TILE_CONFIG['num_tiles']))
    
    throughtput = 1/(total_latency*10**(-3))
    energy_efficiency = throughtput/power

    # Collect intervals per category across ALL accelerators
    emb_intervals   = []
    atten_intervals = []
    fc_intervals    = []
    KV_load_intervals    = []
    others_intervals = []
    link_intervals = []

    for res_name, items in schedule_dict.items():
        for op_name, start, dur in items:
            end = start + dur
            if "->" in op_name:
                link_intervals.append((start, end))
            elif op_name.startswith("emb"):
                emb_intervals.append((start, end))
            elif op_name.startswith("atten"):
                atten_intervals.append((start, end))
            elif op_name.startswith("fc"):
                fc_intervals.append((start, end))
            elif op_name.startswith("KV_load"):
                KV_load_intervals.append((start, end))
            elif op_name.startswith("gelu") or op_name.startswith("layernorm"):
                others_intervals.append((start, end))

    # Union length per category (same units as your schedule_dict times)
    emb_latency_union   = _merged_time_len(emb_intervals)
    atten_latency_union = _merged_time_len(atten_intervals)
    fc_latency_union    = _merged_time_len(fc_intervals)
    KV_load_union    = _merged_time_len(KV_load_intervals)
    link_latency_union    = _merged_time_len(link_intervals)
    others_latency_union    = _merged_time_len(others_intervals)

    # Remaining digital ops = total_latency minus these three unions
    # (Assumes total_latency is the end-to-end makespan in the same units.)
    other_latency = max(0.0, others_latency_union)


    # logging.info(f"Analog energy:                  {ae:.6f} mJ")
    # logging.info(f"Digital Accelerators energy:     {de:.6f} mJ")
    # logging.info(f"Total energy:                   {te:.6f} mJ")
    # logging.info(f"Total avg power:                {power:.6f} W")
    # logging.info(f"Total ACIM tiles used for single tier:   {max_tile_used_single_tier:d} / {ACIM_TILE_CONFIG['num_tiles']}")
    # logging.info(f"Total ACIM tiles used:   {max_tile_used:d} / {ACIM_TILE_CONFIG['num_tiles']*ACIM_TILE_CONFIG['num_tiers']}")
    # logging.info(f"Total PMCA_RED used: {len(used_pmca_red)} / {scheduler.num_pmca_red}")
    # logging.info(f"Total PMCA used: {len(used_pmca)} / {scheduler.num_pmca}")
    # logging.info(f"Total SRAM usage: {sram_usage} MiB. SRAM tiles required: {required_sram_tiles}")
    # logging.info(f"TDP:   {system_TDP:.6f} W")
    # logging.info(f"Throughput: {throughtput:.3f} Inf/s")
    # logging.info(f"Energy efficiency: {energy_efficiency:.3f} Inf/s/W")
    print(f"Analog energy:                 {ae:.6f} mJ")
    print(f"Digital Accelerators energy:    {de:.6f} mJ")
    print(f"Total energy:                  {te:.6f} mJ")
    print(f"Total avg power:               {power:.6f} W")
    print(f"Total area:                    {area_usage:.2f} mm\u00b2")
    print("----------------------------------------------------------------------")
    print(f"Total ACIM tiles used for single tier:   {max_tile_used_single_tier:d} / {ACIM_TILE_CONFIG['num_tiles']}")
    print(f"Total ACIM tiles used:   {max_tile_used:d} / {ACIM_TILE_CONFIG['num_tiles']*ACIM_TILE_CONFIG['num_tiers']}")
    print(f"Total PMCA_RED used: {len(used_pmca_red)} / {scheduler.num_pmca_red}")
    print(f"Total PMCA used: {len(used_pmca)} / {scheduler.num_pmca}")
    print(f"Total DA used: {len(used_da0)} / {scheduler.num_da0}")
    if required_sram_tiles > SRAM_TILE_CONFIG['num_tiles']:
        print(Fore.LIGHTRED_EX + Style.BRIGHT + f"Required SRAM tiles ({required_sram_tiles}) exceed available ({SRAM_TILE_CONFIG['num_tiles']}): DRAM is used." + Style.RESET_ALL)
    else:
        print(f"Total SRAM usage: {sram_usage} MiB. SRAM tiles (of {SRAM_TILE_CONFIG['size']} MiB) required: {required_sram_tiles}")
    #print(f"Total SRAM usage: {sram_usage} MiB. SRAM tiles required: {required_sram_tiles}")
    print(f"TDP:   {system_TDP:.6f} W")
    print("----------------------------------------------------------------------")
    print(Style.BRIGHT +Fore.YELLOW+f"Throughput: {throughtput:.3f} Inf/s")
    print(Style.BRIGHT +Fore.YELLOW+f"Energy efficiency: {energy_efficiency:.3f} Inf/s/W")
    print("----------------------------------------------------------------------")
    # print(f"{Fore.MAGENTA}Total avg power:   {power:.6f} W{Style.RESET_ALL}")

    print(Style.BRIGHT + Fore.CYAN+"\n------------------ Latency Breakdown ------------------"+ Style.RESET_ALL)
    print(f"Embedding                    : {emb_latency_union:.6f}")
    print(f"Attention                    : {atten_latency_union:.6f}")
    print(f"FC                           : {fc_latency_union:.6f}")
    print(f"KV_load                      : {KV_load_union:.6f}")
    print(f"Other digitals               : {other_latency:.6f}")
    print(f"Latency without embeddings   : {total_latency-emb_latency_union:.6f}")
    print(Style.BRIGHT + Fore.CYAN+"-------------------------------------------------------\n" + Style.RESET_ALL)

    print(Fore.LIGHTGREEN_EX + "------------------Energy Breakdown by Type ------------------" + Style.RESET_ALL)
    for t, vals in breakdown_dict["by_type"].items():
        print(f"{t:10s}  Energy: {vals['energy_mJ']:.3f} mJ,  Avg Power: {vals['avg_power_W']:.3f} W")
    print(Style.BRIGHT + Fore.LIGHTGREEN_EX+"---------------------------------------------------------------\n" + Style.RESET_ALL)

    print(Style.BRIGHT +Fore.YELLOW + "\n------------------------ Usage Summary (%) ------------------------\n"+Style.RESET_ALL)

    plot_all_from_existing(accelerators, mapping_summary, summarize_energy_fn=summarize_energy,
                        report_fn=report_acim_digital_link_ddr, display_plots=display_plots, SL=SL, MODEL=model_name)

def clear_logs(model_name):
    log_filename = f"outputs/Attention/{model_name}_attention_tcdm_schedule_chunked_debug.txt"
    os.makedirs("outputs/Attention", exist_ok=True)
    open(log_filename, "w").close() 

def attention_runs_on_DA(schedule_dict,DA_KEY_RE,PMCA_KEY_RE, attn_layer_name: str) -> bool:
    """
    Returns True if any scheduled op for this attention layer is on a DA tile.
    We look for entries whose op name starts with the layer name (common in your logs),
    e.g., 'atten_b0_head0_to_head0' starts with 'atten_b0'.
    """
    for res_name, items in schedule_dict.items():
        for entry in items:
            op_name = entry[0]
            if op_name.startswith(attn_layer_name):
                if DA_KEY_RE.search(res_name):
                    return True
                if PMCA_KEY_RE.search(res_name):
                    return False
    # Default: assume not on DA if not found (conservative)
    return False

CLASSES = ["ACIM", "DAs", "Link", "DDR", "Other"]

def _class_of_name(acc_name: str) -> str:
    n = acc_name.lower()
    if "acimtile" in n: return "ACIM"
    if "pmca" in n or "da" in n: return "DAs"
    if "link" in n: return "Link"
    if "ddr"  in n: return "DDR"
    return "Other"

def _union_len(intervals):
    if not intervals: return 0.0
    iv = sorted(intervals, key=lambda x: x[0])
    merged, cs, ce = [], iv[0][0], iv[0][1]
    for s,e in iv[1:]:
        if s <= ce: ce = max(ce, e)
        else: merged.append((cs, ce)); cs, ce = s, e
    merged.append((cs, ce))
    return sum(e-s for s,e in merged)

def _overlap_len(a_busy, b_busy, T):
    # Simple pairwise bound from busy times; exact interval overlap
    # requires per-class interval intersections. This bound is useful & cheap.
    return max(0.0, a_busy + b_busy - T)

def build_schedule_dict_from_instances(accel_instances):
    return {acc.name: [(ln, float(s), float(d)) for (ln, s, d) in acc.schedule]
            for acc in accel_instances}

def report_acim_digital_link_ddr(accel_instances, analog_pools, summarize_energy_fn, return_dict=False):
    # ---- Build schedule and per-class intervals
    schedule_dict = build_schedule_dict_from_instances(accel_instances)
    per_class_intervals = {c: [] for c in CLASSES}
    per_acc_intervals = {}
    starts, ends = [], []

    for acc_name, ops in schedule_dict.items():
        iv = []
        for name, s, d in ops:
            e = s + d
            iv.append((s, e)); starts.append(s); ends.append(e)
            per_class_intervals[_class_of_name(acc_name)].append((s, e))
        per_acc_intervals[acc_name] = iv

    if not starts:
        print(Fore.RED + "Empty schedule.")
        return {} if return_dict else None

    # ---- Makespan
    T = max(ends) - min(starts)

    # ---- Busy time & utilization
    busy_class = {c: _union_len(per_class_intervals[c]) for c in CLASSES}
    busy_acc   = {a: _union_len(iv) for a, iv in per_acc_intervals.items()}
    util_class = {c: (busy_class[c] / T if T > 0 else 0.0) for c in CLASSES}
    avg_conc   = (sum(busy_acc.values()) / T) if T > 0 else 0.0

    # ---- Overlap summaries (quick, informative)
    ov_AD   = _overlap_len(busy_class["ACIM"],   busy_class["DAs"], T)
    ov_AL   = _overlap_len(busy_class["ACIM"],   busy_class["Link"],    T)
    ov_DDRD = _overlap_len(busy_class["DDR"],    busy_class["DAs"], T)

    # ---- Energy via your function (uses same T for avg power)
    analog_E, digital_E, total_E, avg_P, br = summarize_energy_fn(
        accel_instances, total_lat=T, analog_pools=analog_pools,breakdown=True
    )

    # Aggregate by logical classes from breakdown.by_type (type tags)
    E_class = {c: 0.0 for c in CLASSES}
    for t, entry in br["by_type"].items():
        tl = t.lower()
        e  = entry["energy_mJ"]
        if "acimtile" in tl:        E_class["ACIM"]   += e
        elif "pmca" in tl or "da" in tl: E_class["DAs"]+= e
        elif "link" in tl:          E_class["Link"]   += e
        elif "ddr"  in tl:          E_class["DDR"]    += e
        else:                       E_class["Other"]  += e

    shown_total_E = sum(E_class.values())

    f = lambda x: f"{x:.3f}"

    print(Fore.CYAN +Style.BRIGHT + "\n---------Latency Breakdown (busy-time; util):---------")
    for c in ["ACIM","DAs","Link","DDR"]:
        if busy_class[c] > 0:
            print(f"  {c:<7}: {f(busy_class[c])} ms  (util={util_class[c]*100:.1f}%)")
    if busy_class["Other"] > 0:
        print(f"  Other  : {f(busy_class['Other'])} ms  (util={util_class['Other']*100:.1f})")

    print(f"Average concurrency     : {avg_conc:.2f}")
    if busy_class["ACIM"]>0 and busy_class["DAs"]>0:
        print(f"Overlap ACIM<->DAs    : {f(ov_AD)} ms")
    if busy_class["ACIM"]>0 and busy_class["Link"]>0:
        print(f"Overlap ACIM<->Link       : {f(ov_AL)} ms")
    if busy_class["DDR"]>0 and busy_class["DAs"]>0:
        print(f"Overlap DDR<->DAs     : {f(ov_DDRD)} ms")
    print(Fore.CYAN +Style.BRIGHT+ "----------------------------------------------------------------------\n")

    print(Style.BRIGHT + Fore.LIGHTMAGENTA_EX + "\n---------Energy Breakdown Higher Level---------")
    for c in ["ACIM","DAs","Link","DDR"]:
        if E_class[c] > 0:
            share = 100.0 * E_class[c] / shown_total_E if shown_total_E > 0 else 0.0
            print(f"  {c:<7}: {f(E_class[c])} mJ  ({share:.1f}%)")
    if E_class["Other"] > 0:
        share = 100.0 * E_class["Other"] / shown_total_E if shown_total_E > 0 else 0.0
        print(f"  Other  : {f(E_class['Other'])} mJ  ({share:.1f}%)")
    print(f"  Total  : {f(shown_total_E)} mJ | Avg Power over T: {f(shown_total_E/T if T>0 else 0.0)} W")
    print(Fore.LIGHTMAGENTA_EX +Style.BRIGHT+ "------------------------------------------------\n")
    # # Top-3 busiest accelerators
    # top3 = sorted(busy_acc.items(), key=lambda kv: kv[1], reverse=True)[:3]
    # print("\nTop busy accelerators:")
    # for n, b in top3:
    #     print(f"  * {n:24} {f(b)} ms")

    if return_dict:
        return {
            "T_ms": T,
            "busy_class_ms": busy_class,
            "utilization": util_class,
            "avg_concurrency": avg_conc,
            "overlap_ms": {"ACIM<->DAs": ov_AD, "ACIM<->Link": ov_AL, "DDR<->DAs": ov_DDRD},
            "energy_mJ": E_class,
            "total_energy_mJ": shown_total_E,
            "avg_power_W": shown_total_E / T if T > 0 else 0.0,
        }

# Custom HPCA color palette
COLORS = {
    "ACIM": "#002855",   # dark blue
    "DAs": "#861f41", # dark red
    "Link": "#ffc727",   # gold
    "DDR": "#006341"     # green
}

# ------------------ Busy time vs makespan ------------------
def plot_busy_time_with_makespan(report, title="Busy time per class (union)"):
    order = ["ACIM", "DAs", "Link", "DDR"]
    busy = report["busy_class_ms"]
    T = report["T_ms"]
    util = report["utilization"]

    labels = [c for c in order if c in busy]
    vals = [busy[c] for c in labels]
    colors = [COLORS.get(c, "#999999") for c in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(T, linestyle="--", color="black", linewidth=1.0)
    ax.text(len(labels)-0.5, T, f"T={T:.3f} ms", va="bottom", ha="right", fontsize=11)

    # utilization annotations (x multiplier)
    for i, c in enumerate(labels):
        ax.text(i, vals[i], f"{100*util[c]:.2f}%", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Active time (ms)", fontsize=11)
    #ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    fig.tight_layout()
    return fig, ax


# ------------------ Energy stacked bar ------------------
def plot_energy_stacked(report, title="Energy breakdown by class"):
    order = ["ACIM", "DAs", "Link", "DDR"]
    E = report["energy_mJ"]
    vals = [E.get(c, 0.0) for c in order]
    colors = [COLORS.get(c, "#999999") for c in order]
    total = sum(vals)

    fig, ax = plt.subplots(figsize=(5, 4))
    bottom = 0.0
    for c, v, col in zip(order, vals, colors):
        if v <= 0:
            continue
        ax.bar(["Total"], [v], bottom=bottom, color=col, label=f"{c} ({v:.3f} mJ)",
               edgecolor="black", linewidth=0.7)
        bottom += v

    ax.set_ylabel("Energy (mJ)", fontsize=11)

    if total > 0:
        ax.text(0, total, f"Total={total:.2f} mJ", ha="center", va="bottom", fontsize=11)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    fig.tight_layout()
    return fig, ax

def plot_overlap_heatmap(report, title="Pairwise overlap (ms)"):
    import numpy as np

    classes = ["ACIM", "DAs", "Link", "DDR"]
    busy = report["busy_class_ms"]; T = report["T_ms"]
    present = [c for c in classes if c in busy]
    N = len(present); M = np.zeros((N, N), dtype=float)

    for i, a in enumerate(present):
        for j, b in enumerate(present):
            M[i, j] = busy[a] if i == j else max(0.0, busy[a] + busy[b] - T)

    fig, ax = plt.subplots(figsize=(0.9*N+2, 0.9*N+2))
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(range(N)); ax.set_xticklabels(present, rotation=45, ha="right")
    ax.set_yticks(range(N)); ax.set_yticklabels(present)
    ax.set_title(title)
    for i in range(N):
        for j in range(N):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def plot_all_from_existing(accel_instances, analog_pools, summarize_energy_fn, report_fn, display_plots=False, show_overlap=True, SL=None, MODEL="default"):

    report = report_fn(accel_instances, analog_pools, summarize_energy_fn=summarize_energy_fn, return_dict=True)

    #plot_busy_time_with_makespan(report, title="Busy time per class (union) + makespan")
    #plot_energy_stacked(report, title="Energy breakdown by class")
    # if show_overlap:
    #     plot_overlap_heatmap(report, title="Pairwise overlap (ms)")
    plot_latency_percent(report, "Active time share per accelerator class (%)", SL, MODEL)
    plot_energy_percent(report, "Energy share per accelerator class (%)", SL, MODEL)
    if display_plots:
        plt.show()

def plot_energy_percent(report, title="Energy share per accelerator class (%)", SL=None, model="default"):
    order = ["ACIM", "DAs", "Link", "DDR"]
    E = report["energy_mJ"]
    total = sum(E.get(c, 0.0) for c in order)
    pct = [ (E.get(c, 0.0) / total * 100.0 if total > 0 else 0.0) for c in order ]
    colors = [COLORS.get(c, "#999999") for c in order]

    fig, ax = plt.subplots(figsize=(4, 2))
    bars = ax.bar(order, pct, color=colors, edgecolor="black", linewidth=0.7, width=0.5)

    # annotate percentages on top of bars
    for i, v in enumerate(pct):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Energy\n(% of total energy)", fontsize=11)
    ax.set_ylim(0, max(100, max(pct)*1.1 if pct else 100))
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    # fig.savefig(f"Plots/Breakdown/energy_share_{model}_SL{SL}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"Plots/Breakdown/energy_share_{model}_SL{SL}.svg", format="svg", bbox_inches="tight")
    return fig, ax


def plot_latency_percent(report, title="Active time share per accelerator class (%)", SL=None, model="default"):
    order = ["ACIM", "DAs", "Link", "DDR"]
    busy = report["busy_class_ms"]
    T = report["T_ms"]

    # compute % of total latency (makespan)
    pct = [(busy.get(c, 0.0) / T * 100.0 if T > 0 else 0.0) for c in order]
    colors = [COLORS.get(c, "#999999") for c in order]

    fig, ax = plt.subplots(figsize=(4, 2))
    bars = ax.bar(order, pct, color=colors, edgecolor="black", linewidth=0.7, width=0.5)

    # annotate percentages on top of bars
    for i, v in enumerate(pct):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Active time \n(% of total latency)", fontsize=11)
    ax.set_ylim(0, max(100, max(pct)*1.1 if pct else 100))
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"Plots/Breakdown/latency_share_{model}_SL{SL}.svg", format="svg", bbox_inches="tight")
    return fig, ax