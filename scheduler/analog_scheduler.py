#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#

import networkx as nx
import re
from AnalogMapping import GeneralAnalogTilePool
from nodes.accelerator_config import ACIM_TILE_CONFIG, PMCA_TILE_CONFIG
from nodes.cost_models import MemoryEstimator
from utils.memory_split import LayerSplitter
from utils.chip_utils import dechip, chipify
from utils.layer_splitter_utils import split_layer_intra, get_min_tcdm_capacity_for_op
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import copy
import os
import logging

def premap_analog_layers(graph, analog_pool):
    """
    1) Walk the graph in topological order and extract all analog-type layer names.
    2) Bucket them by their block-suffix (_b0, _b1, _b2, ...).
    3) Allocate every block-0 layer first, then every block-1 layer, etc.
    """

    # 1) collect all analog nodes
    analog_nodes = []
    for node in nx.topological_sort(graph):
        layer = graph.nodes[node]['layer']
        if layer.type in ('fc', 'relu', 'batch_norm', 'residual') :
            analog_nodes.append((node, layer))

    # 2) parse out block index from the name (default to 0 if no _bX suffix)
    def block_idx(name):
        m = re.match(r".+_b(\d+)$", name)
        return int(m.group(1)) if m else 0

    # 3) bucket them by block index
    buckets = {}
    for node, layer in analog_nodes:
        idx = block_idx(layer.name)
        buckets.setdefault(idx, []).append((node, layer))

    # 4) allocate in increasing block order
    for blk in sorted(buckets):
        for node, layer in buckets[blk]:
            # now try-allocate
            if node.startswith('residual'):
                graph.nodes[node]['analog_tiles'] = {}
                graph.nodes[node]['duration']     = None
                graph.nodes[node]['accelerator']  = 'ACIMTile'
                analog_pool.layer_map[node] = {'input_shape': layer.input_shape[0]}
            else:
                placements = analog_pool.try_allocate_layer(
                    layer.name,
                    layer.input_shape[0],
                    layer.output_shape[0]
                )

                # record it on the graph
                graph.nodes[node]['analog_tiles'] = placements
                graph.nodes[node]['duration']     = analog_pool.estimate_latency(layer.name, placements=placements)[0]
                graph.nodes[node]['accelerator']  = 'ACIMTile'

def schedule_model_hybrid(
        graph, 
        accel_instances, 
        mapping_summary, 
        SL=1, 
        pipelining=None, 
        parallelism=None, 
        model_name="default", 
        chunk_optimization=False, 
        chunk_size_d=32, 
        analog_pool=None, 
        use_transfer_link=False,
        display_plots=True,
        digital_opt="latency",
        SL_CROSS=1,
        prefill = None
        ):

    def optimize_chunk_size(graph, SL, display_plots=True, digital_opt="latency"):
        print("[Chunk Optimization Enabled]")
        best_latency = float('inf')
        best_chunk = 32
        coarse_step = 64
        fine_step = 16

        latencies = {}

        print("[Coarse Sweep]")
        coarse_chunks = [8] + [x for x in range(16, SL + 1, coarse_step)]
        if SL not in coarse_chunks:
            coarse_chunks.append(SL)
        for c in coarse_chunks:
            print(f"  Testing chunk size: {c}")
            graph_trial, chunk_size_list = build_chunked_graph(graph, SL=SL, chunk_size=c,mapping_summary=mapping_summary, accelerators=accel_instances, intra_layer_parall=parallelism, analog_pool=analog_pool)
            accel_instances_tmp = copy.deepcopy(accel_instances)
            schedule_dic_tmp, total_latency = schedule_acim_pmca_pipelined_blocks(
                graph_trial, accel_instances_tmp, mapping_summary,
                SL=SL, pipelining=pipelining, parallelism=parallelism,
                model_name=model_name, chunk_size=chunk_size_list,
                chunk_optimization=False,
                analog_pool=analog_pool,
                use_link=use_transfer_link,
                digital_opt=digital_opt,
                SL_CROSS=SL_CROSS,
                prefill=prefill
            )
            latencies[c] = total_latency
            print(f"    -> Latency: {total_latency:.3f} ms")
            if total_latency < best_latency:
                best_latency = total_latency
                best_chunk = c
                best_schedule_dic = schedule_dic_tmp
                best_accel_instances = copy.deepcopy(accel_instances_tmp)

        print("[Fine Sweep]")
        fine_chunks = [x for x in range(max(16, best_chunk - coarse_step), min(SL, best_chunk + coarse_step + 1), fine_step) if x not in latencies]
        for c in fine_chunks:
            print(f"  Testing chunk size: {c}")
            graph_trial, chunk_size_list = build_chunked_graph(graph, SL=SL, chunk_size=c,mapping_summary=mapping_summary, accelerators=accel_instances, intra_layer_parall=parallelism, analog_pool=analog_pool)
            accel_instances_tmp = copy.deepcopy(accel_instances)
            schedule_dic_tmp, total_latency = schedule_acim_pmca_pipelined_blocks(
                graph_trial, accel_instances_tmp, mapping_summary,
                SL=SL, pipelining=pipelining, parallelism=parallelism,
                model_name=model_name, chunk_size=chunk_size_list,
                chunk_optimization=False,
                analog_pool=analog_pool,
                use_link=use_transfer_link,
                digital_opt=digital_opt,
                SL_CROSS=SL_CROSS,
                prefill=prefill
            )
            latencies[c] = total_latency
            print(f"    -> Latency: {total_latency:.3f} ms")
            if total_latency < best_latency:
                best_latency = total_latency
                best_chunk = c
                best_schedule_dic = schedule_dic_tmp
                best_accel_instances = copy.deepcopy(accel_instances_tmp)

        print(f"[Chunk Optimization Result] Best Chunk Size: {best_chunk}, Latency: {best_latency:.3f} ms")

        # Plotting
        if display_plots:
            plt.figure(figsize=(10, 4))
            plt.plot(sorted(latencies.keys()), [latencies[k] for k in sorted(latencies.keys())], marker='o')
            plt.axvline(best_chunk, color='red', linestyle='--', label=f"Best Chunk: {best_chunk}")
            plt.xlabel("Chunk Size")
            plt.ylabel("Latency (ms)")
            plt.title(f"Chunk Size vs Latency ({model_name})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"Plots/{model_name}_chunk_latency_tuning_{pipelining}_pipe_{parallelism}_parall.pdf")
            plt.show()

        return best_schedule_dic, best_latency, best_chunk , best_accel_instances

    def optimize_chunk_size_v2(graph, SL, display_plots=True, digital_opt="latency"):
        print("[Chunk Optimization Enabled]")
        best_latency = float('inf')
        best_chunk = 32
        coarse_step = 64
        fine_step = 16

        latencies = {}

        print("[Coarse Sweep]")
        coarse_chunks = [8] + [x for x in range(16, SL + 1, coarse_step)]
        if SL not in coarse_chunks:
            coarse_chunks.append(SL)
        for c in coarse_chunks:
            print(f"  Testing chunk size: {c}")
            accel_instances_tmp = copy.deepcopy(accel_instances)
            schedule_dic_tmp, total_latency = schedule_block_level_pipeline(
                graph, accel_instances_tmp, mapping_summary,
                SL=SL, pipelining=pipelining, parallelism=parallelism,
                model_name=model_name, chunk_size=c,
                analog_pool=analog_pool,
                use_link=use_transfer_link,
                digital_opt=digital_opt,
                SL_CROSS=SL_CROSS,
                prefill=prefill
                
            )
            latencies[c] = total_latency
            print(f"    -> Latency: {total_latency:.3f} ms")
            if total_latency < best_latency:
                best_latency = total_latency
                best_chunk = c
                best_schedule_dic = schedule_dic_tmp
                best_accel_instances = copy.deepcopy(accel_instances_tmp)

        print("[Fine Sweep]")
        fine_chunks = [x for x in range(max(16, best_chunk - coarse_step), min(SL, best_chunk + coarse_step + 1), fine_step) if x not in latencies]
        for c in fine_chunks:
            print(f"  Testing chunk size: {c}")
            accel_instances_tmp = copy.deepcopy(accel_instances)
            schedule_dic_tmp, total_latency = schedule_block_level_pipeline(
                graph, accel_instances_tmp, mapping_summary,
                SL=SL, pipelining=pipelining, parallelism=parallelism,
                model_name=model_name, chunk_size=c,
                analog_pool=analog_pool,
                use_link=use_transfer_link,
                digital_opt=digital_opt,
                SL_CROSS=SL_CROSS,
                prefill=prefill
            )
            latencies[c] = total_latency
            print(f"    -> Latency: {total_latency:.3f} ms")
            if total_latency < best_latency:
                best_latency = total_latency
                best_chunk = c
                best_schedule_dic = schedule_dic_tmp
                best_accel_instances = copy.deepcopy(accel_instances_tmp)

        print(f"[Chunk Optimization Result] Best Chunk Size: {best_chunk}, Latency: {best_latency:.3f} ms")

        # Plotting
        if display_plots:
            plt.figure(figsize=(10, 4))
            plt.plot(sorted(latencies.keys()), [latencies[k] for k in sorted(latencies.keys())], marker='o')
            plt.axvline(best_chunk, color='red', linestyle='--', label=f"Best Chunk: {best_chunk}")
            plt.xlabel("Chunk Size")
            plt.ylabel("Latency (ms)")
            plt.title(f"Chunk Size vs Latency ({model_name})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"Plots/{model_name}_chunk_latency_tuning_{pipelining}_pipe_{parallelism}_parall.pdf")
            plt.show()

        return best_schedule_dic, best_latency, best_chunk , best_accel_instances
               

    if pipelining in ("inter_layer", "inter_intra_layer"):
        if chunk_optimization:
            schedule_dict, latency, chunk_size, accel_inst = optimize_chunk_size(graph, SL, display_plots, digital_opt=digital_opt) 
            return schedule_dict, latency, chunk_size, accel_inst
        else:
            graph, chunk_size_list = build_chunked_graph(graph, SL=SL, chunk_size=chunk_size_d, mapping_summary=mapping_summary, accelerators=accel_instances, intra_layer_parall=parallelism, analog_pool=analog_pool)
            schedule_dict, latency = schedule_acim_pmca_pipelined_blocks(
                graph, accel_instances, mapping_summary,
                SL=SL, pipelining=pipelining, parallelism=parallelism,
                model_name=model_name, chunk_size=chunk_size_list,
                chunk_optimization=False,
                analog_pool=analog_pool,
                use_link=use_transfer_link,
                digital_opt=digital_opt,
                SL_CROSS=SL_CROSS,
                prefill=prefill
            )
            return schedule_dict, latency, chunk_size_d, accel_instances
    elif pipelining == "inter_block": 
        if chunk_optimization:
            schedule_dict, latency, chunk_size, accel_inst = optimize_chunk_size_v2(graph, SL, display_plots, digital_opt=digital_opt) 
            return schedule_dict, latency, chunk_size, accel_inst
        else:
            schedule_dict, latency =  schedule_block_level_pipeline(
            graph, accel_instances, mapping_summary,
            SL, pipelining, parallelism,
            model_name, chunk_size_d,
            analog_pool=analog_pool,
            use_link=use_transfer_link,
            digital_opt=digital_opt,
            SL_CROSS=SL_CROSS,
            prefill=prefill
        )
        return schedule_dict, latency, chunk_size_d, accel_instances
    else:
        schedule_dict, latency = schedule_model_serial(
            graph, accel_instances, mapping_summary,
            SL=SL, pipelining=pipelining, parallelism=parallelism,
            model_name=model_name,
            analog_pool=analog_pool,
            use_link=use_transfer_link,
            digital_opt=digital_opt,
            SL_CROSS=SL_CROSS,
            prefill=prefill
        )
        return schedule_dict, latency, None, accel_instances



def build_chunked_graph(original_graph, chunk_size, SL, chunkable_types=None,
                        inter_block=False, mapping_summary=None,
                        accelerators=None, digital_opt="latency", #TODO
                        intra_layer_parall=False, analog_pool=None):
    import copy, math, re
    import networkx as nx

    chunked_graph = nx.DiGraph()
    num_chunks = math.ceil(SL / chunk_size)

    if chunkable_types is None:
        chunkable_types = {"fc", "relu", "batch_norm", "gelu", "layernorm", "residual", "gemm"}

    node_chunks = {}           # original node -> [chunk node names]
    chunk_size_list = []
    record_chunk_list = True

    # will store: base_chunk_name -> [parallel split node names] (if any)
    parallel_aliases = {}

    # ---- name parsing (accepts both orders) -------------------------------------
    _RE_CS_1 = re.compile(r'^(?P<base>.*?)(?:_chunk(?P<chunk>\d+))?(?:_split(?P<split>\d+))?$')
    _RE_CS_2 = re.compile(r'^(?P<base>.*?)(?:_split(?P<split>\d+))?(?:_chunk(?P<chunk>\d+))?$')

    def parse_chunk_split(name):
        m = _RE_CS_1.match(name) or _RE_CS_2.match(name)
        if not m:
            return name, None, None
        base = m.group('base')
        chunk = int(m.group('chunk')) if m.group('chunk') is not None else None
        split = int(m.group('split')) if m.group('split') is not None else None
        return base, chunk, split

    # ---- helpers to read shapes from a chunk/split node --------------------------
    def node_rows_cols(g, n, layer_type):
        data = g.nodes[n]
        L = data['layer']
        # Prefer explicit chunk_len for rows
        if layer_type in ('layernorm',):
            if data['layer']:
                rows = data['layer'].input_shape[0]
            else:
                rows = data.get('chunk_len', None)
        else:
            rows = data.get('chunk_len', None)
        if rows is None:
            if hasattr(L, 'input_shape') and L.input_shape:
                rows = L.input_shape[0]
        rows = int(rows) if rows is not None else None

        # Columns depend on type
        if layer_type == 'gelu':
            cols = int(L.input_shape[1])
        elif layer_type == 'atten':
            cols = int(L.output_shape[1])   # heads * head_dim
        elif layer_type in ('layernorm',):
            cols = int(L.input_shape[1])    # hidden dim (constant)
        else:
            cols = None
        return rows, cols

    def expected_rows_cols_original(L, SL):
        t = L.type
        # Expected rows:
        if t in ('layernorm', 'gelu', 'atten', 'embedding', "gemm"):
            exp_rows = int(L.input_shape[0])
        elif t in ('fc', 'residual', 'relu', 'batch_norm'):
            exp_rows = int(SL)
        else:
            exp_rows = None

        # Expected columns for types that split columns:
        if t == 'gelu':
            exp_cols = int(L.input_shape[1])     # hidden dim (e.g., 3072)
        elif t == 'atten':
            exp_cols = int(L.output_shape[1])    # e.g., 12*64 = 768
        elif t == 'layernorm':
            exp_cols = int(L.input_shape[1])     # constant hidden size
        elif t == 'gemm':
            exp_cols = int(L.input_shape[1])  
        else:
            exp_cols = None
        return exp_rows, exp_cols

    # ---- main validator ----------------------------------------------------------
    def validate_full_coverage(original_graph, chunked_graph, SL, chunkable_types, verbose=True):
        """
        Checks per original layer:
        1) Row coverage across chunks: sum(rows) over chunks == expected rows.
        2) Within each chunk, depending on layer type:
            - layernorm: sum(rows across splits) == chunk_len, and each split cols == hidden dim.
            - gelu: each split rows == chunk_len; sum(cols across splits) == original hidden dim.
            - atten: each split rows == chunk_len; sum(output cols across splits) == original heads*head_dim.
            - fc/residual/relu/bn: only row coverage across chunks (no column split expected).

        Prints issues and returns a problems dict (empty if all good).
        """
        problems = {}

        for onode in original_graph.nodes():
            L = original_graph.nodes[onode]['layer']
            base = L.name
            t = L.type
            if not any(t in base for t in chunkable_types):             
                continue

            exp_rows, exp_cols = expected_rows_cols_original(L, SL)

            # --- group chunked nodes for this base by chunk index
            chunks = {}  # chunk_idx -> [node names]
            for n in chunked_graph.nodes():
                b, ch, sp = parse_chunk_split(n)
                if b == base and ch is not None:
                    chunks.setdefault(ch, []).append(n)

            if not chunks:
                # some layers may remain unchunked by design; skip strict check
                continue

            # --- 1) Row coverage across chunks
            row_sum = 0
            missing_chunks = []
            max_ch = max(chunks.keys())
            margin_ln = 0
            for i in range(max_ch + 1):
                if i not in chunks:
                    missing_chunks.append(i)
                else:
                    # Take any rep: all splits in a chunk share the same chunk_len rows,
                    # except for layernorm row-splitting where splits partition rows.
                    rep = chunks[i][0]
                    rows_rep, _ = node_rows_cols(chunked_graph, rep, t)
                    # For LN: rows may be re-split across splits; compute actual sum across splits
                    if t == 'layernorm' and len(chunks[i]) > 1:
                        rows_in_chunk = 0
                        for n in chunks[i]:
                            r, _ = node_rows_cols(chunked_graph, n, t)
                            rows_in_chunk += (r or 0)
                            margin_ln += 1
                        row_sum += rows_in_chunk
                    else:
                        # GeLU/Atten splits columns -> each split has same rows == chunk_len
                        row_sum += (rows_rep or 0)

            if exp_rows is not None:
                if (row_sum > exp_rows + margin_ln) or (row_sum < exp_rows) or missing_chunks:
                    problems.setdefault(base, {})['rows'] = {
                        'expected': exp_rows, 'sum': row_sum, 'missing_chunks': missing_chunks
                    }
                    if verbose:
                        miss = f", missing_chunks={missing_chunks}" if missing_chunks else ""
                        print(f"[ROW] {base}: expected={exp_rows}, sum={row_sum}{miss}")


            # --- 2) Per-chunk column/row invariants depending on type
            if t == 'gelu' and exp_cols is not None:
                margin = 0
                # GeLU: within each chunk: sum of split cols == exp_cols AND each split rows == chunk_len
                for ch_idx, nodes in chunks.items():
                    # find chunk_len from any split
                    r0, _ = node_rows_cols(chunked_graph, nodes[0], t)
                    # sum columns across splits
                    col_sum = 0
                    bad_rows = []
                    for n in nodes:
                        r, c = node_rows_cols(chunked_graph, n, t)
                        col_sum += (c or 0)
                        margin += 1
                        if r != r0:
                            bad_rows.append((n, r, r0))
                    if (col_sum > exp_cols + margin) or (col_sum < exp_cols) or bad_rows:
                        problems.setdefault(base, {}).setdefault('gelu_cols', {})[ch_idx] = {
                            'expected_cols': exp_cols, 'sum_cols': col_sum, 'row_mismatch': bad_rows
                        }
                        if verbose:
                            print(f"[COL] {base} chunk{ch_idx}: expected_cols={exp_cols}, sum_cols={col_sum}")
                            for n, r, rref in bad_rows:
                                print(f"      rows mismatch in {n}: got {r}, expected {rref}")

        if verbose and not problems:
            print("[CHECK] All row/column coverage checks passed.")
        return problems

    # ---------- helpers ----------
    def expand(name):
        """Return [name] or its parallel children if it was split."""
        return parallel_aliases.get(name, [name])

    def add_parallel_children_for_chunk(base_name, chunked_graph, chunk_layer, chunk_idx, accel_instances):
        """
        Replace a single chunk node with multiple split nodes (intra-layer parallelism).
        Wires will be handled later via expand().
        """
        # Compute the parallel split layers
        split_layers = split_layer_intra(chunk_layer, accel_instances, digital_opt)
        if len(split_layers) <= 1:
            return  # nothing to do

        # Remove the base node and create split nodes
        preds = list(chunked_graph.predecessors(base_name))
        succs = list(chunked_graph.successors(base_name))
        chunk_len = chunked_graph.nodes[base_name].get("chunk_len", None)

        chunked_graph.remove_node(base_name)

        split_names = []
        for idx, sublayer in enumerate(split_layers):
            split_name = f"{base_name}_split{idx}"
            chunked_graph.add_node(
                split_name,
                layer=sublayer,
                chunk_idx=chunk_idx,
                chunk_len=chunk_len,
                # keep any other attrs you need
            )
            split_names.append(split_name)

        # Reconnect: preds -> all splits, and all splits -> succs
        for p in preds:
            for s in split_names:
                chunked_graph.add_edge(p, s)
        for s in split_names:
            for q in succs:
                chunked_graph.add_edge(s, q)

        parallel_aliases[base_name] = split_names

    def adjust_layer_memory_for_node(graph, node):
        """
        Check memory for this node's layer and, if too large, attach intra_layer_splits.
        No renaming of nodes -- consistent with check_and_adjust_layer_memory().
        """
        L = graph.nodes[node]['layer']

        if L.type == "layernorm":
            cap_kb = get_min_tcdm_capacity_for_op("layernorm", accelerators)
            mem_kb = MemoryEstimator.estimate_memory_kb(L, flash_attention=False)
            if mem_kb > cap_kb:
                L.force_serial = True
                L.intra_layer_splits = LayerSplitter.split_sublayer_to_fit_tcdm(L, cap_kb)

        elif L.type == "gelu":
            cap_kb = get_min_tcdm_capacity_for_op("gelu", accelerators)
            mem_kb = MemoryEstimator.estimate_memory_kb(L, flash_attention=False)
            if mem_kb > cap_kb:
                L.force_serial = True
                L.intra_layer_splits = LayerSplitter.split_sublayer_to_fit_tcdm(L, cap_kb)

        elif L.type == "gemm":
            cap_kb = get_min_tcdm_capacity_for_op("gemm", accelerators)
            mem_kb = MemoryEstimator.estimate_memory_kb(L, flash_attention=False)
            if mem_kb > cap_kb:
                L.force_serial = True
                L.intra_layer_splits = LayerSplitter.split_sublayer_to_fit_tcdm(L, cap_kb)


    # ---------- 1) create chunked nodes ----------
    for node in nx.topological_sort(original_graph):
        layer = original_graph.nodes[node]['layer']
        name = layer.name

        if layer.type in chunkable_types:
            node_chunks[node] = []
            for i in range(num_chunks):
                this_len = chunk_size if i < num_chunks - 1 else SL - chunk_size * (num_chunks - 1)

                chunk_layer = copy.deepcopy(layer)
                chunk_name = f"{name}_chunk{i}"

                if layer.type in ("gelu", "layernorm", "gemm"):
                    # adjust shapes for digital chunk
                    r, c = chunk_layer.input_shape
                    chunk_layer.input_shape = (this_len, c)
                    if hasattr(chunk_layer, "output_shape"):
                        ro, co = chunk_layer.output_shape
                        chunk_layer.output_shape = (this_len, co)

                chunked_graph.add_node(
                    chunk_name,
                    layer=chunk_layer,
                    chunk_idx=i,
                    chunk_len=this_len,
                )
                node_chunks[node].append(chunk_name)

                # optional serial dependency within a layer over its chunks
                if i > 0 and layer.type not in ("layernorm", "gelu", "gemm"):
                    chunked_graph.add_edge(node_chunks[node][i - 1], chunk_name)

                if record_chunk_list:
                    chunk_size_list.append(this_len)
            record_chunk_list = False

        else:
            # non-chunkable (atten or others you treat as full)
            chunked_graph.add_node(name, layer=layer, chunk_idx=0, chunk_len=SL)
            node_chunks[node] = [name]

    # ---------- 2) apply intra-layer parallelism inside chunks ----------
    if intra_layer_parall:
        for node in list(chunked_graph.nodes()):
            L = chunked_graph.nodes[node]['layer']
            if L.type in ("layernorm", "gelu", "gemm"):
                # only parallelize if this chunk is >1 row (your requirement)
                if chunked_graph.nodes[node].get("chunk_len", 1) > 1 or (L.type in ("gelu", "gemm")):
                    add_parallel_children_for_chunk(
                        base_name=node,
                        chunked_graph=chunked_graph,
                        chunk_layer=L,
                        chunk_idx=chunked_graph.nodes[node]["chunk_idx"],
                        accel_instances=accelerators
                    )

    # ---------- 2.5) memory-fit splits for LN/GeLU (after chunking & intra-parallel) ----------
    # Walk over *current* nodes (including any _split* children) and split those that overflow memory.
    for node in list(chunked_graph.nodes()):
        L = chunked_graph.nodes[node]['layer']
        if L.type in ("layernorm", "gelu", "gemm"):
            adjust_layer_memory_for_node(chunked_graph, node)

    # ---------- 3) wire edges from original graph, expanding if needed ----------
    for src, dst in original_graph.edges():
        src_chunks = node_chunks[src]
        dst_chunks = node_chunks[dst]

        src_layer = original_graph.nodes[src]['layer']
        dst_layer = original_graph.nodes[dst]['layer']

        if dst_layer.type == "atten":
            # Attention depends on last chunk of each input
            for u in expand(src_chunks[-1]):
                for v in expand(dst_chunks[0]):
                    chunked_graph.add_edge(u, v)

        elif src_layer.type == "atten":
            # Post-attention layers depend on full attention output
            for dst_chunk in dst_chunks:
                for u in expand(src_layer.name):
                    for v in expand(dst_chunk):
                        chunked_graph.add_edge(u, v)

        elif src_layer.type in chunkable_types and dst_layer.type in chunkable_types:
            # chunk i -> chunk i (expand to splits if present)
            for i in range(num_chunks):
                for u in expand(src_chunks[i]):
                    for v in expand(dst_chunks[i]):
                        chunked_graph.add_edge(u, v)

        elif dst_layer.type in chunkable_types and src_layer.type not in chunkable_types:
            # broadcast scalar/full to all chunks (and their splits)
            for dst_chunk in dst_chunks:
                for u in expand(src_layer.name):
                    for v in expand(dst_chunk):
                        chunked_graph.add_edge(u, v)

        else:
            # default single edge
            for u in expand(src_layer.name if src_layer.type not in chunkable_types else src_chunks[0]):
                for v in expand(dst_layer.name if dst_layer.type not in chunkable_types else dst_chunks[0]):
                    chunked_graph.add_edge(u, v)

    # ---------- 4) your inter_block barrier logic ----------
    if not inter_block:
        block_to_nodes = {}
        block_id_pattern = re.compile(r"_b(\d+)(?:_|$)")
        #for n in chunked_graph.nodes():
        for n in nx.topological_sort(chunked_graph):
            m = block_id_pattern.search(n)
            if m:
                blk = int(m.group(1))
                block_to_nodes.setdefault(blk, []).append(n)

        all_blocks = sorted(block_to_nodes.keys())
        for i in all_blocks:
            if i == 0:
                continue
            prev_blk = i - 1
            if prev_blk not in block_to_nodes:
                continue

            # keep your existing barrier logic (optionally expand through aliases if needed)
            if len(block_to_nodes)>1 and prev_blk==0:
                num_KV_loads = sum(1 for count in block_to_nodes[1] if count.startswith("KV_load"))
                cnt_last_layers_to_skip = 1 if len(block_to_nodes[0])==len(block_to_nodes[1]) else 1+abs((len(block_to_nodes[0])-len(block_to_nodes[1])+num_KV_loads))
            else:
                cnt_last_layers_to_skip = 1
            src_node = block_to_nodes[prev_blk][-cnt_last_layers_to_skip]
            block_to_nodes_without_KV_load = [s for s in block_to_nodes[i] if not s.startswith("KV_load")]
            shared_inputs_list = []
            match_name = re.search(r'.*_b\d+', block_to_nodes_without_KV_load[0])
            dst_name = match_name.group(0) if match_name else None
            tmp_list = [dst_name]
            for shared_l in analog_pool.share_map:
                if shared_l in block_to_nodes_without_KV_load[0]:
                    shared_inputs_list = analog_pool.share_map[shared_l]
                    tmp_list.append(shared_l)
                    break
            if len(shared_inputs_list)> 0:
                tmp_list.extend(list(shared_inputs_list))
            tmp_list.append("KV_load")
            updated_tmp_list = [re.sub(r'_b\d+', '_b'+str(i), list_up) for list_up in tmp_list]
            input_to_add_dep = [single_block for single_block in block_to_nodes[i] if any(inp in single_block for inp in updated_tmp_list)]
            for dst_node in input_to_add_dep:#block_to_nodes[i][0:final_num_chunk+num_KV_loads]:
                for u in expand(src_node):
                    for v in expand(dst_node):
                        if not chunked_graph.has_edge(u, v):
                            chunked_graph.add_edge(u, v)

    problems = validate_full_coverage(original_graph, chunked_graph,SL, chunkable_types,verbose=True)
    if problems:
        raise RuntimeError("Generated chunked graph has size mismatches:", problems)
    return chunked_graph, chunk_size_list

def schedule_acim_pmca_pipelined_blocks(
        graph, 
        accel_instances, 
        mapping_summary, 
        SL=1, 
        pipelining=None, 
        parallelism=None, 
        model_name="default", 
        chunk_size=[32],
        chunk_optimization=False,
        analog_pool=None,
        use_link=False,
        digital_opt="latency",
        SL_CROSS=None,
        prefill=None
        ):
    
    # --- setup logger ---
    log_dir = "outputs/Scheduling"
    os.makedirs(log_dir, exist_ok=True)
    fname = f"INFO_scheduling_{model_name}_SL{SL}_{pipelining or 'none'}_pipe_{parallelism or 'none'}_parall_chunk{chunk_size[0]}{'_opt' if chunk_optimization else ''}.log"
    log_path = os.path.join(log_dir, fname)
    logger = logging.getLogger(f"{model_name}_pipelined")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [fh]
    
    def scale_analog_latency(base_latency, SL):
        return base_latency * SL

    def check_analog_conflicts(start, dur, slots, col_blocks, mapping_summary, accel_by_name, num_tiers, chip_id=0):
        """Check conflicts across tiers and overlapping column blocks."""
        bumped_start = start
        for tile_id, tier_id in slots:
            #inst_name = f"ACIMTile[{tile_id},{tier_id}]"
            inst_name = chipify(f"ACIMTile[{tile_id},{tier_id}]", chip_id)
            inst = accel_by_name[inst_name]

            # Same-tile, other-tier conflicts
            for other_t in range(num_tiers):
                if other_t == tier_id:
                    continue 
                other_inst = accel_by_name.get(chipify(f"ACIMTile[{tile_id},{other_t}]", chip_id))
                if not other_inst:
                    continue
                for _, s2, d2 in other_inst.schedule:
                    if not (s2 + d2 <= bumped_start or bumped_start + dur <= s2):
                        bumped_start = s2 + d2
                        return bumped_start, True

            # Same-tier, column-overlap conflicts
            for other, s2, d2 in inst.schedule:
                if not (s2 + d2 <= bumped_start or bumped_start + dur <= s2): #and not other.startswith("residual"):
                    for (c0a, c1a) in col_blocks[tile_id]:
                        col_bks_other = mapping_summary.get(other, {}).get('col_blocks', {}).get(tile_id, [])
                        for (c0b, c1b) in col_bks_other: # mapping_summary[other]['col_blocks'][tile_id]:
                            if not (c1a <= c0b or c1b <= c0a):
                                bumped_start = s2 + d2
                                return bumped_start, True
            # Residual special case: check conflicts with all other residuals on same tile/tier
            for other, s2, d2 in inst.schedule:
                if other.startswith("residual") and not (s2 + d2 <= bumped_start or bumped_start + dur <= s2):
                    bumped_start = s2 + d2
                    return bumped_start, True
        return bumped_start, False
    
    def find_earliest_non_overlapping_start(schedule, t_ready, ready_time, latency):
        """
        Find the earliest start time >= max(t_ready, acc_ready_time) where
        [start, start+latency) does not overlap any scheduled intervals.
        Assumes schedule is sorted by start time.
        """
        if not schedule:
            return t_ready

        t_start = max(t_ready, ready_time)

        # Sort the schedule just in case
        sorted_schedule = sorted(schedule, key=lambda x: x[1])

        for i, (_, s_start, s_latency) in enumerate(sorted_schedule):
            s_end = s_start + s_latency

            if t_start + latency <= s_start:
                # We can schedule before this interval
                return t_start

            # If t_start overlaps, move to end of this interval
            if t_start < s_end:
                t_start = s_end

            # Now check for a gap *between* this interval and the next one
            if i + 1 < len(sorted_schedule):
                _, next_start, _ = sorted_schedule[i + 1]
                if t_start + latency <= next_start:
                    return t_start

        # If no suitable gap was found, schedule after the last interval
        return t_start
    
    def pick_best(scored_cands, opt):
        if opt == "latency":
            best = min(scored_cands, key=lambda x: (x[0], x[1]))
        elif opt == "power":
            best = min(scored_cands, key=lambda x: (x[1], x[0]))
        else:
            raise ValueError(f"Unsupported optimisation target: {opt}. Supported digital optimisation targets are 'latency' and 'power'.")
        return best 

    def pick_digital_accelerator(layer, t_ready, accel_instances, pipelining=None, parallelism=None, inter_layer_chunk_size=None, latency_cache=None, chunk_ready_time_cache=None, model_name="default", chunk_idx=None, digital_opt="latency"):
        cands = [a for a in accel_instances if a.can_run(layer) and not dechip(a.name).startswith("Link[")]

        if latency_cache is None:
            latency_cache = {}
        if chunk_ready_time_cache is None:
            chunk_ready_time_cache = {}

        latencies_by_type = {}
        chunk_ready_times_by_type = {}

        for acc in cands:
            acc_type = dechip(acc.name).split('[')[0]
            latency_key = (layer.type, pipelining, parallelism, acc_type, inter_layer_chunk_size, layer.input_shape, layer.output_shape, layer.attention_type)
            chunk_ready_time_key = (layer.type, pipelining, parallelism, acc_type, inter_layer_chunk_size)


            if latency_key not in latency_cache:
                print(f"Latency estimation for the {acc.type_tag} accelerator type: (chunk_size={inter_layer_chunk_size})")
                result = acc.estimate_cost(layer, pipelining=pipelining, parallelism=parallelism, inter_layer_chunk_size=inter_layer_chunk_size)

                if isinstance(result, tuple):
                    latency, chunk_ready_times = result
                    if layer.type == "atten":
                        if chunk_ready_time_key in chunk_ready_time_cache:
                            # Merge with existing entries
                            existing_times = chunk_ready_time_cache[chunk_ready_time_key]
                            for k, v in chunk_ready_times.items():
                                if v > existing_times[k]:
                                    chunk_ready_time_cache[chunk_ready_time_key][k] = v
                        else:
                            chunk_ready_time_cache[chunk_ready_time_key] = chunk_ready_times
                    else:
                        chunk_ready_time_cache[chunk_ready_time_key] = chunk_ready_times
                else:
                    latency = result
                    chunk_ready_time_cache[chunk_ready_time_key] = {}

                latency_cache[latency_key] = latency
                print(f"{latency:.3f}")
            
            latencies_by_type[acc.name] = latency_cache[latency_key]
            chunk_ready_times_by_type[acc.name] = chunk_ready_time_cache.get(chunk_ready_time_key, {})

        scored_cands = []
        for acc in cands:
            latency = latencies_by_type[acc.name]
            chunk_ready_times = chunk_ready_times_by_type.get(acc.name, {})
            ready_time = chunk_ready_times.get(chunk_idx, 0.0) if isinstance(chunk_ready_times, dict) and len(chunk_ready_times)>1 else 0.0

            start_time = find_earliest_non_overlapping_start(acc.schedule, t_ready, ready_time, latency)
            end_time = start_time + latency
            power = acc.estimate_power(layer)
            scored_cands.append((end_time, power, latency, acc, ready_time))

        best_end_time, best_power, latency, best_accel, ready_time = pick_best(scored_cands,digital_opt)
        return best_accel, latency, ready_time
    
    def compute_ready_time(node, scheduled, graph, chunk_ready_time_cache=None, chunk_size=None, link = 0):
        """
        Compute when a node's input data is ready, based on predecessors.
        For attention -> FC pipelining, adjusts t_ready using chunked latency.
        """
        import numpy as np

        t_ready = 0.0
        chunk_idx = graph.nodes[node].get("chunk_idx", None)
        chunk_idx_tmp = 0
        new_chunk_size = chunk_size

        for pred in graph.predecessors(node):
            if link:
                m_chunk_pred = re.search(r'_chunk(\d+)$', pred)
                m_chunk_node = re.search(r'_chunk(\d+)$', node)
                if m_chunk_pred and m_chunk_node:
                    base_name_pred    = pred[:m_chunk_pred.start()]
                    base_name_node    = node[:m_chunk_node.start()]
                    if base_name_pred == base_name_node:
                        continue
            #hasattr(layer, "intra_layer_splits")
            pred_start, pred_dur, _ = scheduled.get(pred, (0.0, 0.0, None))
            pred_end = pred_start + pred_dur
            pred_type = graph.nodes[pred].get("layer", {}).type if "layer" in graph.nodes[pred] else None

            # grab the Layer object for "pred"
            pred_layer = graph.nodes[pred].get("layer", None)
            if pred_layer is not None and hasattr(pred_layer, "intra_layer_splits"):
                num_split = len(pred_layer.intra_layer_splits)
                inp_size = pred_layer.input_shape[0]
                num_chunk_split = inp_size/num_split/chunk_size
                current_split = int(chunk_idx//num_chunk_split)
                pred_start = pred_start + current_split * pred_dur/num_split
                chunk_idx_tmp = int(chunk_idx%num_chunk_split)
            # Special case: attention -> fc chunk
            if (
                chunk_ready_time_cache
                and chunk_idx is not None
                and pred_type == "atten"
            ):
                # Try any matching latency key
                for (ltype, _, _, _, csize), chunk_dict in chunk_ready_time_cache.items():
                    if ltype == "atten" and csize == chunk_size:
                        if pred_layer is not None and hasattr(pred_layer, "intra_layer_splits"):
                            latency = chunk_dict.get(chunk_idx_tmp)
                        else:
                            latency = chunk_dict.get(chunk_idx)
                        if latency is not None:
                            pred_end = pred_start + float(latency)
                            break

            t_ready = max(t_ready, pred_end)

        return t_ready#, new_chunk_size

    def split_info(s):
        # returns (base_name, split_number)
        m = re.match(r'(.*?)(?:_split(\d+))?$', s)
        base = m.group(1)
        num  = int(m.group(2)) if m.group(2) is not None else -1
        return base, num
    
    accel_by_name = {a.name: a for a in accel_instances}
    chip_id = 0
    link_inst = next(a for a in accel_instances if dechip(a.name).startswith("Link[")) #ADDED
    all_slots = [slot for summ in mapping_summary.values() for slot in summ["slots"]]
    num_tiers = max(t for _, t in all_slots) + 1 if all_slots else 1

    scheduled = {}
    schedule_dict = {acc.name: [] for acc in accel_instances}
    latency_cache = {}
    chunk_ready_time_cache= {}
    ready_time=0.0
    transfer_cache = {}
    seen_atten = []

    logger.info(f"Scheduling chunk size: {chunk_size[0]}")

    for node in nx.topological_sort(graph):
        layer = graph.nodes[node]['layer']
        chunk_idx = graph.nodes[node].get("chunk_idx", None)
        name = layer.name
        chunk = chunk_size[chunk_idx] if chunk_idx else chunk_size[0]
        t_ready = compute_ready_time(
            node,
            scheduled,
            graph,
            chunk_ready_time_cache,
            chunk
        )

        # attention-block sync
        #m = re.match(r"(atten_b\d+)", name)
        m = re.match(r"(atten\d*_b\d+)", name)
        prefix = m.group(1) if m else None
        t_ready_link = t_ready
        # ---------------------- LINK TRANSFERS ----------------------
        if use_link:
            #already_explored_atten = False
            already_explored_atten = []
            if node.startswith("KV_load"):
                t_ready = t_ready_link
                if node not in transfer_cache or t_ready != transfer_cache[node]:
                    transfer_cache[node] = t_ready
            else:
                for pred in graph.predecessors(node):
                    if pred.startswith("KV_load"):
                        logger.info(f"[LINK] skip splitted layer {pred}->{node} (DMA takes care of it).")
                        t_ready = max(t_ready,t_ready_link)
                        continue
                    layer_pred = graph.nodes[pred]['layer']
                    name_pred = layer_pred.name
                    if name_pred == name:
                        t_ready_link = compute_ready_time(
                            node,
                            scheduled,
                            graph,
                            chunk_ready_time_cache,
                            chunk,
                            link=1
                        )
                        continue
                    m_chunk = re.search(r'_chunk(\d+)', node)
                    if m_chunk:
                        # m_chunk.group(0) == '_chunkN'
                        chunk_suffix = m_chunk.group(0)
                        # m_chunk.group(1) == 'N' (just the digits)
                        chunk_id     = m_chunk.group(1)
                        # everything *before* '_chunkN'
                        base_name    = name[:m_chunk.start()]
                    else:
                        chunk_suffix = None
                        chunk_id     = None
                        base_name    = name

                    # Skip redundant moves when pred & node share inputs
                    if analog_pool and hasattr(analog_pool, "share_map"):
                        sm = analog_pool.share_map
                        # if preceed by attention only take one of them
                        if pred.startswith("atten") or pred.startswith("gelu") or pred.startswith("layernorm"):
                            if pred in already_explored_atten:
                                continue
                            else:
                                already_explored_atten.append(pred)

                        # do they share?
                        if name in sm:
                            skip_pred = False
                            existing_node = None
                            # both must be analog-placed
                            for shared_node in sm[name]:
                                #if name_pred in sm:
                                if shared_node+chunk_suffix in transfer_cache:  #and mapping_summary[name]["tiles_used"]==mapping_summary[shared_node]["tiles_used"]:
                                    if name.startswith("residual") and pred.startswith("fc"):
                                        continue
                                    logger.info(f"[LINK] skip shared-input {pred}->{name}.")
                                    skip_pred = True
                                    existing_node = shared_node+chunk_suffix
                                    break
                            if skip_pred:
                                t_ready = max(transfer_cache[existing_node],t_ready_link)
                                continue
                    if layer.type == "residual":
                        continue
                    if prefix and prefix in seen_atten:
                        t_ready = transfer_cache[prefix]
                        continue

                    # when did producer finish?
                    if layer_pred.type == "atten":
                        prod_finish = t_ready_link
                    else :
                        st_pred, du_pred, _ = scheduled[pred]
                        prod_finish = st_pred + du_pred
                    
                    # ask Link how long to move pred's output
                    # (CostModel.link_latencycost reads layer.output_shape)
                    if "split" in node and not "split0" in node:
                        prefix = node.split('_split')[0]
                        if node.startswith("atten"):
                            match_suff = re.search(r"(_chunk\d+)$", node)
                            suffix = match_suff.group(1) if match_suff else ""    
                            logger.info(f"[LINK] skip splitted layer {pred}->{node}.")
                            t_ready = transfer_cache[prefix+"_split0"+suffix]
                            break
                        logger.info(f"[LINK] skip splitted layer {pred}->{node}.")
                        t_ready = transfer_cache[prefix+"_split0"]
                        break
                    elif pred.startswith("atten") == False and "split" in pred : #and not "split0" in pred:
                        if (pred.startswith("gelu")==True and "_r" in pred) and not "split0" in pred:
                            prefix = pred.split('_split')[0]    
                            logger.info(f"[LINK] skip splitted layer {pred}->{node}.")
                            t_ready = transfer_cache[node]
                            break
                        else:
                            xfer_dur = link_inst.estimate_cost(graph.nodes[pred]['layer'], SL_scale=graph.nodes[pred]['layer'].input_shape[0])
                    
                    else:
                        if node.startswith("atten"):
                            if pred.startswith("fc") and graph.nodes[pred]['layer'].attention_type == "cross":
                                if prefill <=1:
                                    xfer_dur = link_inst.estimate_cost(graph.nodes[pred]['layer'],SL_scale=SL_CROSS)
                                else:
                                    xfer_dur = 0.0
                            else:
                                xfer_dur = link_inst.estimate_cost(graph.nodes[pred]['layer'], SL_scale=SL)
                        else:
                            xfer_dur = link_inst.estimate_cost(graph.nodes[pred]['layer'], SL_scale=chunk)

                    busy = sorted([(st, st+dur) for (_, st, dur) in link_inst.schedule], key=lambda x: x[0])
                    t = prod_finish
                    for bs, be in busy:
                        if t + xfer_dur <= bs:
                            break
                        t = max(t, be)
                    start = t
                    end = start + xfer_dur
                    # optional: record it
                    if node.startswith("atten"):
                        p = re.search(r'_chunk(\d+)$', pred)
                        pred_prefix= pred[:p.start()]
                        link_inst.schedule.append((f"{pred_prefix}->{node}", start, xfer_dur))
                        schedule_dict[link_inst.name].append((f"{pred_prefix if pred_prefix else pred}->{node}", start, xfer_dur))
                        logger.info(f"[LINK] transfer {pred_prefix if pred_prefix else pred}->{node} on {link_inst.name} @ {start:.6f} ms, dur={xfer_dur:.3f} ms")
                    else:
                        skip = False
                        if "chunk" in pred:
                            for scheduled_inst in schedule_dict[link_inst.name]:
                                if f"{pred}->" in scheduled_inst[0]:
                                    logger.info(f"[LINK] Skip {pred}->{node} on {link_inst.name}, already transferred {scheduled_inst[0]}")
                                    skip = True
                                    break
                        if not skip:
                            link_inst.schedule.append((f"{pred}->{node}", start, xfer_dur))
                            schedule_dict[link_inst.name].append((f"{pred}->{node}", start, xfer_dur))
                            logger.info(f"[LINK] transfer {pred}->{node} on {link_inst.name} @ {start:.6f} ms, dur={xfer_dur:.6f} ms")
                    # bump t_ready so compute only starts after the last transfer
                    t_ready = max(t_ready,end)#, xfer_start + xfer_dur)
                    if node not in transfer_cache or t_ready != transfer_cache[node]:
                        transfer_cache[node] = t_ready

        if prefix and use_link and prefix not in seen_atten:
            seen_atten.append(prefix)
            transfer_cache[prefix] = t_ready

        # -------- Intra-layer split handling (already preprocessed) --------
        if hasattr(layer, "intra_layer_splits"):
            cur_time = t_ready
            for sub in layer.intra_layer_splits:
                best, dur, ready_time = pick_digital_accelerator(sub, cur_time, accel_instances, 
                                                                 pipelining=pipelining, 
                                                                 parallelism=parallelism,
                                                                 latency_cache=latency_cache,
                                                                 chunk_ready_time_cache=chunk_ready_time_cache,
                                                                 model_name=model_name,
                                                                 inter_layer_chunk_size=chunk,
                                                                 chunk_idx=chunk_idx,
                                                                 digital_opt=digital_opt
                                                                 )   
                start = find_earliest_non_overlapping_start(best.schedule, cur_time, cur_time, dur)
                best.available_time = start + dur
                best.schedule.append((sub.name, start, dur))
                schedule_dict[best.name].append((sub.name, start, dur))
                cur_time = start + dur
                logger.info(f"Scheduled layer split {sub.name:6s} on {best.name:20s} @ {start:.6f} dur={dur:.6f}")
            
            scheduled[node] = (t_ready, cur_time - t_ready, best.name)
            logger.info(f"Scheduled {name:6s} on {best.name:20s} @ {t_ready:.6f} dur={cur_time - t_ready:.6f}")
            continue

        if layer.type in ("fc", "relu", "batch_norm"):
            slots = mapping_summary[name]['slots']
            base_dur = mapping_summary[name]['latency']
            col_blocks = mapping_summary[name]['col_blocks']

            if layer.attention_type == "cross" and prefill <= 1:
                dur = scale_analog_latency(base_dur, SL_CROSS)
            elif layer.attention_type == "cross" and prefill > 1:
                dur = 0.0
            else:
                dur = scale_analog_latency(base_dur, chunk)
            start = t_ready
            
            while True:
                new_start, bumped = check_analog_conflicts(start, dur, slots, col_blocks, mapping_summary, accel_by_name, num_tiers, chip_id)
                if bumped:
                    start = new_start
                else:
                    break

            acc_names = []
            for tile_id, tier_id in slots:
                inst_name = chipify(f"ACIMTile[{tile_id},{tier_id}]", chip_id)
                inst = accel_by_name[inst_name]
                inst.schedule.append((name, start, dur))
                schedule_dict[inst_name].append((name, start, dur))
                acc_names.append(inst_name)
            scheduled[node] = (start, dur, ",".join(acc_names))

        elif layer.type == "residual":
            #print(f"[PPU Residual] Processing {name}")
            logger.info(f"[PPU Residual] Processing {name}")
            src_layers = list(graph.predecessors(node))
            src_layers_pred_chunk = list(graph.predecessors(node))
            src_layers.sort(key=lambda n: scheduled[n][0], reverse=True)
            not_from_analog = False
            src_layers = [re.sub(r"_chunk\d+$", "", src) for src in src_layers]            # Find analog-mapped tiles from predecessors
            
            # 2) record base-order as they appear
            base_order = []
            for s in src_layers:
                b, _ = split_info(s)
                if b not in base_order:
                    base_order.append(b)
            base_index = {b:i for i,b in enumerate(base_order)}

            # 4) final sort key: by base's original index, then by split number
            src_layers = sorted(
                src_layers,
                key=lambda s: (
                    base_index[split_info(s)[0]],
                    split_info(s)[1]
                )
            )

            mapped_tiles = []
            for src in src_layers:
                if src in mapping_summary:
                    mapped_tiles.extend(mapping_summary[src]["slots"])

            if not mapped_tiles:
                logger.info(f"[PPU Residual] no ACIM predecessor for {name}. TMP")
                not_from_analog = True
                for analog_layer in mapping_summary:
                    if analog_layer.startswith("fc"):
                        mapped_tiles.extend(mapping_summary[analog_layer]["slots"])
                # raise RuntimeError(f"No analog-mapped input found for residual {name}")

            m_chunk = re.search(r'_chunk(\d+)$', node)
            chunk_suffix = m_chunk.group(0)
            # Try scheduling on each mapped tile
            assigned = False
            for tile_id, tier_id in mapped_tiles:
                inst_name = chipify(f"ACIMTile[{tile_id},{tier_id}]", chip_id)
                inst = accel_by_name[inst_name]

                # ---- 1) LINK transfers for preds NOT on this (tile_id,tier_id) ----
                #local_ready = t_ready
                if use_link:
                    for src in src_layers:
                        # if this src *is* on our candidate tile, skip the link hop
                        if (tile_id, tier_id) in mapping_summary.get(src, {}).get("slots", []):
                            logger.info(f"[LINK] skip ext-tile-acc {src}->{name} on {inst_name}")
                            continue
                        if (layer.name in sm and not src.startswith("fc")) or src.startswith("residual"):
                            continue                      
                        chunk_suffix_pred = [s for s in src_layers_pred_chunk if s.startswith(src)]
                        # otherwise, schedule a LINK transfer
                        st_pred, du_pred, _ = scheduled[chunk_suffix_pred[0]]
                        prod_finish = st_pred + du_pred

                        if src.startswith("atten") == False and "split" in src :#and not "split0" in pred:
                            if (src.startswith("gelu")==True and "_r" in src) and not "split0" in src:
                                prefix = src.split('_split')[0] 
                                suffix = src.split('_split')[1]   
                                if suffix in node:
                                    logger.info(f"[LINK] skip splitted layer {src}->{node}.")
                                    t_ready = transfer_cache[node]
                                break
                            else:
                                xfer_dur = link_inst.estimate_cost(graph.nodes[chunk_suffix_pred[0]]['layer'], SL_scale=graph.nodes[chunk_suffix_pred[0]]['layer'].input_shape[0]) # chunk
                        else:
                            xfer_dur   = link_inst.estimate_cost(graph.nodes[chunk_suffix_pred[0]]['layer'], SL_scale=chunk)

                        
                        busy = sorted([(st, st+dur) for (_, st, dur) in link_inst.schedule], key=lambda x: x[0])
                        t = prod_finish
                        for bs, be in busy:
                            if t + xfer_dur <= bs:
                                break
                            t = max(t, be)
                        start = t
                        end = start + xfer_dur
                        link_inst.schedule.append((f"{chunk_suffix_pred[0]}->{node}", start, xfer_dur))
                        schedule_dict[link_inst.name].append((f"{chunk_suffix_pred[0]}->{node}", start, xfer_dur))
                        logger.info(f"[LINK] transfer {chunk_suffix_pred[0]}->{node} on {link_inst.name} "
                                    f"@ {start:.6f} ms, dur={xfer_dur:.6f} ms")
                        t_ready = max(t_ready,end)
                        transfer_cache[node] = t_ready

                # Check if this tile was also used by any predecessor -> enables ext_tile_acc
                ext_tile_acc = any(
                    (tile_id, tier_id) in mapping_summary.get(src, {}).get("slots", [])
                    for src in src_layers
                )

                # Estimate latency using GeneralAnalogTilePool API
                latency = analog_pool.estimate_latency_residual(
                    layer_size=layer.input_shape[0], SL=chunk, ext_tile_acc=ext_tile_acc, not_from_analog=not_from_analog
                )

                start = t_ready
                while True:
                    res_conflict_end = min(
                        (s + d for lname, s, d in inst.schedule
                        if lname.startswith("residual")
                        and not (s + d <= start or start + latency <= s)),
                        default=None
                    )
                    if not_from_analog:
                        fc_conflict_end = min(
                            (s + d for lname, s, d in inst.schedule
                            if lname.startswith("fc")
                            and not (s + d <= start or start + latency <= s)),
                            default=None
                        )
                    else: 
                        fc_conflict_end = None
                    res_conflict = res_conflict_end is not None
                    fc_conflict = fc_conflict_end is not None
                    if res_conflict or fc_conflict:
                        start = max(e for e in (res_conflict_end, fc_conflict_end) if e is not None) #0.0001  # step forward slightly to resolve conflict
                    else:
                        break

                # Schedule residual on PPU
                inst.schedule.append((name, start, latency))
                schedule_dict[inst_name].append((name, start, latency))
                scheduled[node] = (start, latency, inst_name)

                logger.info(f"Scheduled residual {name} on {inst_name} @ {start:.6f} dur={latency:.6f} ms (ext_tile_acc={ext_tile_acc}, not_from_analog={not_from_analog})")
                assigned = True
                break

            if not assigned:
                raise RuntimeError(f"No available PPU found for residual {name}")

            continue
        else:
            best, dur, ready_time = pick_digital_accelerator(
                layer, t_ready, accel_instances,
                pipelining=pipelining, parallelism=parallelism,
                latency_cache=latency_cache,
                chunk_ready_time_cache=chunk_ready_time_cache,
                model_name=model_name,
                inter_layer_chunk_size=chunk,
                chunk_idx=chunk_idx,
                digital_opt=digital_opt
            )
            start = find_earliest_non_overlapping_start(best.schedule, t_ready, ready_time or 0.0, dur)
            best.available_time = start + dur
            best.schedule.append((name, start, dur))
            schedule_dict[best.name].append((node, start, dur))
            scheduled[node] = (start, dur, best.name)

        st, du, an = scheduled[node]
        graph.nodes[node].update({"accelerator": an, "start_time": st, "duration": du})
        logger.info(f"Scheduled {name:6s} on {an:20s} @ {st:.6f} dur={du:.6f}")

    starts = [st for st, _, _ in scheduled.values()]
    ends = [st + dur for st, dur, _ in scheduled.values()]
    logger.info(f"[Pipeline Schedule Complete (chunk size {chunk_size[0]})] total latency: {max(ends) - min(starts):.6f} ms")

    return schedule_dict, max(ends) - min(starts)



def schedule_model_serial(graph, accel_instances, mapping_summary, SL=1, pipelining=None, parallelism=None, model_name="default", analog_pool=None, use_link=False, digital_opt="latency", SL_CROSS=None, prefill = None):
    # --- setup logger ---
    log_dir = "outputs/Scheduling"
    os.makedirs(log_dir, exist_ok=True)
    fname = f"INFO_scheduling_{model_name}_SL{SL}_serial_{pipelining or 'none'}_pipe_{parallelism or 'none'}.log"
    log_path = os.path.join(log_dir, fname)
    logger = logging.getLogger(f"{model_name}_serial")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [fh]
    
    def compute_ready_time(node, scheduled, pipelining_ratio=1.0):
        """Compute when a node's input data is ready."""
        t_ready = 0.0
        for pred in graph.predecessors(node):
            st, du, _ = scheduled[pred]
            t_ready = max(t_ready, st + du * pipelining_ratio)
        return t_ready

    def scale_analog_latency(base_latency, SL):
        return base_latency * SL

    def check_analog_conflicts(start, dur, slots, col_blocks, mapping_summary, accel_by_name, num_tiers, chip_id=0):
        """Check conflicts across tiers and overlapping column blocks."""
        bumped_start = start
        for tile_id, tier_id in slots:
            #inst_name = f"ACIMTile[{tile_id},{tier_id}]"
            inst_name = chipify(f"ACIMTile[{tile_id},{tier_id}]", chip_id)
            inst = accel_by_name[inst_name]

            # Same-tile, other-tier conflicts
            for other_t in range(num_tiers):
                if other_t == tier_id:
                    continue 
                other_inst = accel_by_name.get(chipify(f"ACIMTile[{tile_id},{other_t}]", chip_id)) 
                if not other_inst:
                    continue
                for _, s2, d2 in other_inst.schedule:
                    if not (s2 + d2 <= bumped_start or bumped_start + dur <= s2):
                        bumped_start = s2 + d2
                        return bumped_start, True

            # Same-tier, column-overlap conflicts
            for other, s2, d2 in inst.schedule:
                if not (s2 + d2 <= bumped_start or bumped_start + dur <= s2):
                    for (c0a, c1a) in col_blocks[tile_id]:
                        col_bks_other = mapping_summary.get(other, {}).get('col_blocks', {}).get(tile_id, [])
                        for (c0b, c1b) in col_bks_other: #mapping_summary[other]['col_blocks'][tile_id]:
                            if not (c1a <= c0b or c1b <= c0a):
                                bumped_start = s2 + d2
                                return bumped_start, True

            # Residual special case: check conflicts with all other residuals on same tile/tier
            for other, s2, d2 in inst.schedule:
                if other.startswith("residual") and not (s2 + d2 <= bumped_start or bumped_start + dur <= s2):
                    bumped_start = s2 + d2
                    return bumped_start, True

        return bumped_start, False

    def find_earliest_non_overlapping_start(schedule, t_ready, ready_time, latency):
        """
        Find the earliest start time >= max(t_ready, acc_ready_time) where
        [start, start+latency) does not overlap any scheduled intervals.
        Assumes schedule is sorted by start time.
        """
        if not schedule:
            return t_ready

        t_start = max(t_ready, ready_time)

        # Sort the schedule just in case
        sorted_schedule = sorted(schedule, key=lambda x: x[1])

        for i, (_, s_start, s_latency) in enumerate(sorted_schedule):
            s_end = s_start + s_latency

            if t_start + latency <= s_start:
                # We can schedule before this interval
                return t_start

            # If t_start overlaps, move to end of this interval
            if t_start < s_end:
                t_start = s_end

            # Now check for a gap *between* this interval and the next one
            if i + 1 < len(sorted_schedule):
                _, next_start, _ = sorted_schedule[i + 1]
                if t_start + latency <= next_start:
                    return t_start

        # If no suitable gap was found, schedule after the last interval
        return t_start
    
    def pick_best(scored_cands, opt):
        if opt == "latency":
            best = min(scored_cands, key=lambda x: (x[0], x[1]))
        elif opt == "power":
            best = min(scored_cands, key=lambda x: (x[1], x[0]))
        else:
            raise ValueError(f"Unsupported optimisation target: {opt}. Supported digital optimisation targets are 'latency' and 'power'.")
        return best 
    
    def pick_digital_accelerator(layer, t_ready, accel_instances, pipelining=None, parallelism=None, latency_cache=None, model_name="default",digital_opt="latency"):
        cands = [a for a in accel_instances if a.can_run(layer) and not dechip(a.name).startswith("Link[") ]

        if latency_cache is None:
            latency_cache = {}

        # Key by (layer type, pipelining, parallelism, accelerator type)
        latencies_by_type = {}
        for acc in cands:
            acc_type = dechip(acc.name).split('[')[0]  # e.g., "PMCA" or "PMCA_RED"
            latency_key = (layer.type, pipelining, parallelism, acc_type, layer.input_shape, layer.output_shape, layer.attention_type)

            if latency_key not in latency_cache:
                print(f"Latency estimation for the {acc.type_tag} accelerator type")
                latency = acc.estimate_cost(layer, pipelining=pipelining, parallelism=parallelism)
                latency_cache[latency_key] = latency

            latencies_by_type[acc.name] = latency_cache[latency_key]

        # Now score each candidate using its specific latency
        scored_cands = []
        for acc in cands:
            latency = latencies_by_type[acc.name]
            start_time = find_earliest_non_overlapping_start(acc.schedule, t_ready, t_ready, latency)
            end_time = start_time + latency
            power = acc.estimate_power(layer)
            scored_cands.append((end_time, power, latency, acc))

        best_end_time, best_power, latency, best_accel = pick_best(scored_cands,digital_opt)
        return best_accel, latency
    
    def ensure_block_connection(graph, analog_pool=None):
        block_to_nodes = {}
        block_id_pattern = re.compile(r"_b(\d+)(?:_|$)")
        #for n in chunked_graph.nodes():
        for n in nx.topological_sort(graph):
            m = block_id_pattern.search(n)
            if m:
                blk = int(m.group(1))
                block_to_nodes.setdefault(blk, []).append(n)

        all_blocks = sorted(block_to_nodes.keys())
        for i in all_blocks:
            if i == 0:
                continue
            prev_blk = i - 1
            if prev_blk not in block_to_nodes:
                continue

            # keep your existing barrier logic (optionally expand through aliases if needed)
            if len(block_to_nodes)>1 and prev_blk==0:
                num_KV_loads = sum(1 for count in block_to_nodes[1] if count.startswith("KV_load"))
                cnt_last_layers_to_skip = 1 if len(block_to_nodes[0])==len(block_to_nodes[1]) else 1+abs((len(block_to_nodes[0])-len(block_to_nodes[1])+num_KV_loads))
            else:
                cnt_last_layers_to_skip = 1
            src_node = [block_to_nodes[prev_blk][-cnt_last_layers_to_skip]]
            block_to_nodes_without_KV_load = [s for s in block_to_nodes[i] if not s.startswith("KV_load")]
            shared_inputs_list = []
            match_name = re.search(r'.*_b\d+', block_to_nodes_without_KV_load[0])
            dst_name = match_name.group(0) if match_name else None
            tmp_list = [dst_name]
            for shared_l in analog_pool.share_map:
                if shared_l in block_to_nodes_without_KV_load[0]:
                    shared_inputs_list = analog_pool.share_map[shared_l]
                    break
            if len(shared_inputs_list)> 0:
                tmp_list.extend(list(shared_inputs_list))
            tmp_list.append("KV_load")
            updated_tmp_list = [re.sub(r'_b\d+', '_b'+str(i), list_up) for list_up in tmp_list]
            input_to_add_dep = [single_block for single_block in block_to_nodes[i] if any(inp in single_block for inp in updated_tmp_list)]
            for v in input_to_add_dep:#[0:final_num_chunk+num_KV_loads]:
                for u in src_node:
                    if not graph.has_edge(u, v):
                        graph.add_edge(u, v)
        return graph

    # ---------------------- Main Scheduler ----------------------
    accel_by_name = {acc.name: acc for acc in accel_instances}
    chip_id = 0
    link_inst = next(a for a in accel_instances if dechip(a.name).startswith("Link[")) #ADDED
    all_slots = [slot for summ in mapping_summary.values() for slot in summ["slots"]]
    num_tiers = max(t for _, t in all_slots) + 1 if all_slots else 1

    scheduled = {}
    schedule_dict = {acc.name: [] for acc in accel_instances}

    latency_cache = {}
    transfer_cache = {}  # Cache for transfer latencies
    seen_atten = []

    graph = ensure_block_connection(graph, analog_pool)

    for node in nx.topological_sort(graph):
        layer = graph.nodes[node]['layer']
        name = layer.name
        pipelining_ratio = 1.0 #0.25 if pipelining else 1.0
        t_ready = compute_ready_time(node, scheduled, pipelining_ratio)

        m = re.match(r"(atten\d*_b\d+)", name)
        prefix = m.group(1) if m else None

        # ---------------------- LINK TRANSFERS ----------------------
        if use_link:
            already_explored_atten = [] #False
            if node.startswith("KV_load"):
                if node not in transfer_cache or t_ready != transfer_cache[node]:
                    transfer_cache[node] = t_ready
            else:
                for pred in graph.predecessors(node):
                    layer_pred = graph.nodes[pred]['layer']
                    name_pred = layer_pred.name
                    # Skip redundant moves when pred & node share inputs
                    if pred.startswith("KV_load"):
                            logger.info(f"[LINK] skip splitted layer {pred}->{node} (DMA takes care of it).")
                            t_ready = t_ready
                            continue
                    if analog_pool and hasattr(analog_pool, "share_map"):
                        sm = analog_pool.share_map
                        # if preceed by attention only take one of them
                        if pred.startswith("atten") or pred.startswith("gelu") or pred.startswith("layernorm"):
                            if pred in already_explored_atten:
                                continue
                            else:
                                already_explored_atten.append(pred)
                        # do they share?
                        if node in sm:
                            skip_pred = False
                            existing_node = None
                            # both must be analog-placed
                            for shared_node in sm[node]:
                                #if name_pred in sm:
                                if shared_node in transfer_cache: # and mapping_summary[node]["tiles_used"]==mapping_summary[shared_node]["tiles_used"]:
                                    if node.startswith("residual") and pred.startswith("fc"):
                                        continue
                                    logger.info(f"[LINK] skip shared-input {pred}->{node}.")
                                    skip_pred = True
                                    existing_node = shared_node
                                    break
                            if skip_pred:
                                t_ready = max(transfer_cache[existing_node],t_ready)
                                continue
                    if layer.type == "residual":
                        continue
                    if prefix and prefix in seen_atten:
                        t_ready = transfer_cache[prefix]
                        continue

                    # when did producer finish?
                    st_pred, du_pred, _ = scheduled[pred]
                    prod_finish = st_pred + du_pred * pipelining_ratio

                    # ask Link how long to move pred's output
                    # (CostModel.link_latencycost reads layer.output_shape)
                    if "split" in node and not "split0" in node:
                        prefix = node.split('_split')[0]    
                        logger.info(f"[LINK] skip splitted layer {pred}->{node}.")
                        t_ready = transfer_cache[prefix+"_split0"]
                        break
                    elif pred.startswith("atten") == False and "split" in pred: # and not "split0" in pred:
                        if (pred.startswith("gelu")==True and "_r" in pred) and not "split0" in pred:
                            prefix = pred.split('_split')[0]    
                            logger.info(f"[LINK] skip splitted layer {pred}->{node}.")
                            t_ready = transfer_cache[node]
                            break
                        else:
                            xfer_dur = link_inst.estimate_cost(graph.nodes[pred]['layer'], SL_scale=graph.nodes[pred]['layer'].input_shape[0]) #SL
                    else:
                        if pred.startswith("fc") and graph.nodes[pred]['layer'].attention_type == "cross":
                            if prefill <= 1:
                                xfer_dur = link_inst.estimate_cost(graph.nodes[pred]['layer'],SL_scale=SL_CROSS)
                            else:
                                xfer_dur = 0.0
                        else:
                            xfer_dur = link_inst.estimate_cost(graph.nodes[pred]['layer'],SL_scale=SL)

                    # schedule it--must wait both for prod_finish and link availability
                    xfer_start = max(prod_finish, link_inst.available_time)
                    link_inst.available_time = xfer_start + xfer_dur
                    # optional: record it
                    skip = False
                    if "chunk" in pred:
                        for scheduled_inst in schedule_dict[link_inst.name]:
                            if f"{pred}->" in scheduled_inst[0]:
                                logger.info(f"[LINK] Skip {pred}->{node} on {link_inst.name}, already transferred {scheduled_inst[0]}")
                                skip = True
                                break
                    if not skip:
                        link_inst.schedule.append((f"{pred}->{node}", xfer_start, xfer_dur))
                        schedule_dict[link_inst.name].append((f"{pred}->{node}", xfer_start, xfer_dur))
                        logger.info(f"[LINK] transfer {pred}->{node} on {link_inst.name} @ {xfer_start:.3f} ms, dur={xfer_dur:.6f} ms")

                    # bump t_ready so compute only starts after the last transfer
                    t_ready = max(t_ready, xfer_start + xfer_dur)
                    if node not in transfer_cache or transfer_cache[node] != t_ready:
                        transfer_cache[node] = t_ready

        if prefix and use_link and prefix not in seen_atten:
            seen_atten.append(prefix)
            transfer_cache[prefix] = t_ready

        # -------- Intra-layer split handling (already preprocessed) --------
        if hasattr(layer, "intra_layer_splits"):
            cur_time = t_ready
            for sub in layer.intra_layer_splits:
                best, dur = pick_digital_accelerator(sub, cur_time, accel_instances, pipelining, parallelism, latency_cache, model_name, digital_opt=digital_opt)
                start = find_earliest_non_overlapping_start(best.schedule, cur_time, cur_time, dur)
                best.available_time = start + dur
                best.schedule.append((sub.name, start, dur))
                schedule_dict[best.name].append((sub.name, start, dur))
                cur_time = start + dur
                logger.info(f"Scheduled layer split {sub.name:6s} on {best.name:20s} @ {start:.3f} dur={dur:.6f}")
            
            scheduled[node] = (t_ready, cur_time - t_ready, best.name)
            logger.info(f"Scheduled {name:6s} on {best.name:20s} @ {t_ready:.3f} dur={cur_time - t_ready:.6f}")
            continue

        # ================= Handle Standard (non-parallel) scheduling =================
        if layer.type in ("fc", "relu", "batch_norm"):
            # -------- Analog Execution --------
            slots = mapping_summary[name]['slots']
            base_dur = mapping_summary[name]['latency']
            col_blocks = mapping_summary[name]['col_blocks']

            #if pipelining is None and parallelism is None:
            if layer.attention_type == "cross" and prefill <= 1:
                dur = scale_analog_latency(base_dur, SL_CROSS)
            elif layer.attention_type == "cross" and prefill > 1:
                dur = 0.0
            else:
                dur = scale_analog_latency(base_dur, SL)


            start = t_ready

            while True:
                new_start, bumped = check_analog_conflicts(start, dur, slots, col_blocks, mapping_summary, accel_by_name, num_tiers, chip_id)
                if bumped:
                    start = new_start
                else:
                    break

            acc_names = []
            for tile_id, tier_id in slots:
                inst_name = chipify(f"ACIMTile[{tile_id},{tier_id}]", chip_id)
                inst = accel_by_name[inst_name]
                inst.schedule.append((name, start, dur))
                schedule_dict[inst_name].append((name, start, dur))
                acc_names.append(inst_name)

            scheduled[node] = (start, dur, ",".join(acc_names))

        elif layer.type == "residual":
            logger.info(f"[PPU Residual] Processing {name}")
            src_layers = list(graph.predecessors(node))
            src_layers.sort(key=lambda n: scheduled[n][0], reverse=True)
            not_from_analog = False
            # Find analog-mapped tiles from predecessors
            mapped_tiles = []
            for src in src_layers:
                if src in mapping_summary:
                    if src.startswith("residual"):
                        src_layers_int = list(graph.predecessors(src))
                        for src2 in src_layers_int:
                            if src2 in mapping_summary:
                                mapped_tiles.extend(mapping_summary[src2]["slots"])
                    else:
                        mapped_tiles.extend(mapping_summary[src]["slots"])

            if not mapped_tiles:
                logger.info(f"[PPU Residual] no ACIM predecessor for {name}. TMP")
                not_from_analog = True
                for analog_layer in mapping_summary:
                    if analog_layer.startswith("fc"):
                        mapped_tiles.extend(mapping_summary[analog_layer]["slots"])

            # Try scheduling on each mapped tile
            assigned = False
            for tile_id, tier_id in mapped_tiles:
                inst_name = chipify(f"ACIMTile[{tile_id},{tier_id}]", chip_id)
                inst = accel_by_name[inst_name]

                # ---- 1) LINK transfers for preds NOT on this (tile_id,tier_id) ----
                local_ready = t_ready
                if use_link:
                    for src in src_layers:
                        # if this src *is* on our candidate tile, skip the link hop
                        if (tile_id, tier_id) in mapping_summary.get(src, {}).get("slots", []):
                            logger.info(f"[LINK] skip ext-tile-acc {src}->{name} on {inst_name}")
                            continue
                        if layer.name in sm and not src.startswith("fc"):
                            continue 
                        # otherwise, schedule a LINK transfer
                        st_pred, du_pred, _ = scheduled[src]
                        prod_finish = st_pred + du_pred * pipelining_ratio

                        if src.startswith("atten") == False and "split" in src and not "split0" in src:
                            if src.startswith("gelu")==False or (src.startswith("gelu")==True and "_r" in src):
                                prefix = src.split('_split')[0]    
                                logger.info(f"[LINK] skip splitted layer {src}->{name}.")
                                break
                            else:
                                xfer_dur   = link_inst.estimate_cost(graph.nodes[src]['layer'], SL_scale=SL)
                        else:
                            xfer_dur   = link_inst.estimate_cost(graph.nodes[src]['layer'], SL_scale=SL)

                        xfer_start = max(prod_finish, link_inst.available_time)
                        link_inst.available_time = xfer_start + xfer_dur
                        link_inst.schedule.append((f"{src}->{name}", xfer_start, xfer_dur))
                        schedule_dict[link_inst.name].append((f"{src}->{name}", xfer_start, xfer_dur))
                        logger.info(f"[LINK] transfer {src}->{name} on {link_inst.name} "
                                    f"@ {xfer_start:.3f} ms, dur={xfer_dur:.4f} ms")

                        local_ready = max(local_ready, xfer_start + xfer_dur)
                        transfer_cache[node] = local_ready


                # Check if this tile was also used by any predecessor -> enables ext_tile_acc
                ext_tile_acc = any(
                    (tile_id, tier_id) in mapping_summary.get(src, {}).get("slots", [])
                    for src in src_layers
                )

                # Estimate latency using GeneralAnalogTilePool API
                latency = analog_pool.estimate_latency_residual(
                    layer_size=layer.input_shape[0], SL=SL, ext_tile_acc=ext_tile_acc, not_from_analog=not_from_analog
                )

                start = local_ready
                while True:
                    res_conflict_end = min(
                        (s + d for lname, s, d in inst.schedule
                        if lname.startswith("residual")
                        and not (s + d <= start or start + latency <= s)),
                        default=None
                    )
                    if not_from_analog:
                        fc_conflict_end = min(
                            (s + d for lname, s, d in inst.schedule
                            if lname.startswith("fc")
                            and not (s + d <= start or start + latency <= s)),
                            default=None
                        )
                    else:
                        fc_conflict_end = None
                    res_conflict = res_conflict_end is not None
                    fc_conflict = fc_conflict_end is not None
                    if res_conflict or fc_conflict:
                        start = max(e for e in (res_conflict_end, fc_conflict_end) if e is not None) #0.0001  # step forward slightly to resolve conflict
                    else:
                        break

                # Schedule residual on PPU
                inst.schedule.append((name, start, latency))
                schedule_dict[inst_name].append((name, start, latency))
                scheduled[node] = (start, latency, inst_name)

                #print(f"Scheduled residual {name} on {inst_name} @ {start:.3f} dur={latency:.6f} (ext_tile_acc={ext_tile_acc})")
                logger.info(f"Scheduled residual {name} on {inst_name} @ {start:.3f} dur={latency:.6f} ms (ext_tile_acc={ext_tile_acc}, not_from_analog={not_from_analog})")
                assigned = True
                break

            if not assigned:
                raise RuntimeError(f"No available PPU found for residual {name}")

            continue
        else:
            best, dur = pick_digital_accelerator(layer, t_ready, accel_instances, pipelining, parallelism, latency_cache, model_name, digital_opt=digital_opt)
            start = find_earliest_non_overlapping_start(best.schedule, t_ready, t_ready, dur)
            best.available_time = start + dur
            best.schedule.append((name, start, dur))
            schedule_dict[best.name].append((name, start, dur))
            scheduled[node] = (start, dur, best.name)
                
        # === Common update and print (for both analog and digital) ===
        st, du, an = scheduled[node]
        graph.nodes[node].update({
            "accelerator": an,
            "start_time": st,
            "duration": du
        })
        logger.info(f"Scheduled {name:6s} on {an:20s} @ {st:.3f} dur={du:.6f}")
                
    # Final latency report
    starts = [st for st, _, _ in scheduled.values()]
    ends = [st + dur for st, dur, _ in scheduled.values()]
    logger.info(f"[Serial Schedule Complete] total latency: {max(ends) - min(starts):.6f} ms")
    return schedule_dict, max(ends) - min(starts)


def schedule_block_level_pipeline(
    graph,
    accel_instances,
    mapping_summary,
    SL,
    pipelining,
    parallelism,
    model_name,
    chunk_size,
    analog_pool=None,
    use_link=False,
    digital_opt="latency",
    SL_CROSS=None, 
    prefill = None
):
    
        # --- setup logger ---
    log_dir = "outputs/Scheduling"
    os.makedirs(log_dir, exist_ok=True)
    fname = f"INFO_scheduling_block_level_{model_name}_SL{SL}_{pipelining or 'none'}_pipe_{parallelism or 'none'}_parall_chunk{chunk_size}.log"
    log_path = os.path.join(log_dir, fname)
    logger = logging.getLogger(f"{model_name}_pipelined")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [fh]
    
    """
    Level-0 pipelining within each block via schedule_acim_pmca_pipelined_blocks,
    then block-level pipelining across blocks by carrying forward each tile's
    available_time.
    """
        # --- 2) identify last-per-block in the *original* graph ---
    #      bucket original-nodes by block-index from their name suffix "_bX"

    chunked_graph, chunk_size_list = build_chunked_graph(graph, SL=SL, chunk_size=chunk_size, inter_block=True, mapping_summary=mapping_summary, accelerators=accel_instances, intra_layer_parall=parallelism, analog_pool=analog_pool)

    blocks = defaultdict(list)
    for n in nx.topological_sort(chunked_graph):
        m = re.search(r"_b(\d+)", chunked_graph.nodes[n]['layer'].name)
        b = int(m.group(1)) if m else 0
        blocks[b].append(n)
    # now extract the *last* layer-name per block
    last_of_block = {
        b: blocks[b][-1]  # assumes sorted by topo
        for b in blocks
    }

    def scale_analog_latency(base_latency, SL):
        return base_latency * SL

    def check_analog_conflicts(start, dur, slots, col_blocks, mapping_summary, accel_by_name, num_tiers, share_map, layer_name, chip_id=0):
        """Check conflicts across tiers and overlapping column blocks."""
        bumped_start = start
        bumped = start
        for tile_id, tier_id in slots:
            inst_name = chipify(f"ACIMTile[{tile_id},{tier_id}]", chip_id)
            inst = accel_by_name[inst_name]

            # Same-tile, other-tier conflicts
            for other_t in range(num_tiers):
                if other_t == tier_id:
                    continue   
                other_inst = accel_by_name.get(chipify(f"ACIMTile[{tile_id},{other_t}]", chip_id))
                if not other_inst:
                    continue
                for _, s2, d2 in other_inst.schedule:
                    if not (s2 + d2 <= bumped_start or bumped_start + dur <= s2):
                        bumped_start = s2 + d2
                        return bumped_start, True

            # Same-tier, column-overlap conflicts
            for other, s2, d2 in inst.schedule:
                if not (s2 + d2 <= bumped_start or bumped_start + dur <= s2):
                    for (c0a, c1a) in col_blocks[tile_id]:
                        col_bks_other = mapping_summary.get(other, {}).get('col_blocks', {}).get(tile_id, [])
                        for (c0b, c1b) in col_bks_other:
                            if not (c1a <= c0b or c1b <= c0a):
                                bumped_start = s2 + d2
                                return bumped_start, True
            

            # 2) Same-tier: only allow overlap if inputs are shared
            for other_name, s2, d2 in inst.schedule:
                # if no temporal overlap, fine
                if s2 + d2 <= bumped or bumped + dur <= s2:
                    continue

                # if these two layers share inputs, we allow concurrency
                if layer_name in share_map and other_name in share_map[layer_name]:
                    continue

                # otherwise we must bump
                bumped = s2 + d2
                return bumped, True

            # Residual special case: check conflicts with all other residuals on same tile/tier
            for other, s2, d2 in inst.schedule:
                if other.startswith("residual") and not (s2 + d2 <= bumped_start or bumped_start + dur <= s2):
                    bumped_start = s2 + d2
                    return bumped_start, True
            
        return bumped_start, False
    
    def find_earliest_non_overlapping_start(schedule, t_ready, ready_time, latency):
        """
        Find the earliest start time >= max(t_ready, acc_ready_time) where
        [start, start+latency) does not overlap any scheduled intervals.
        Assumes schedule is sorted by start time.
        """
        if not schedule:
            return t_ready

        t_start = max(t_ready, ready_time)

        # Sort the schedule just in case
        sorted_schedule = sorted(schedule, key=lambda x: x[1])

        for i, (_, s_start, s_latency) in enumerate(sorted_schedule):
            s_end = s_start + s_latency

            if t_start + latency <= s_start:
                # We can schedule before this interval
                return t_start

            # If t_start overlaps, move to end of this interval
            if t_start < s_end:
                t_start = s_end

            # Now check for a gap *between* this interval and the next one
            if i + 1 < len(sorted_schedule):
                _, next_start, _ = sorted_schedule[i + 1]
                if t_start + latency <= next_start:
                    return t_start

        # If no suitable gap was found, schedule after the last interval
        return t_start

    def pick_best(scored_cands, opt):
        if opt == "latency":
            best = min(scored_cands, key=lambda x: (x[0], x[1]))
        elif opt == "power":
            best = min(scored_cands, key=lambda x: (x[1], x[0]))
        else:
            raise ValueError(f"Unsupported optimisation target: {opt}. Supported digital optimisation targets are 'latency' and 'power'.")
        return best 

    def pick_digital_accelerator(layer, t_ready, accel_instances, pipelining=None, parallelism=None, inter_layer_chunk_size=None, latency_cache=None, chunk_ready_time_cache=None, model_name="default", chunk_idx=None, digital_opt="latency"):
        cands = [a for a in accel_instances if a.can_run(layer) and not dechip(a.name).startswith("Link[") and not dechip(a.name).startswith("ACIM")]

        if latency_cache is None:
            latency_cache = {}
        if chunk_ready_time_cache is None:
            chunk_ready_time_cache = {}

        latencies_by_type = {}
        chunk_ready_times_by_type = {}

        for acc in cands:
            acc_type = dechip(acc.name).split('[')[0]
            latency_key = (layer.type, pipelining, parallelism, acc_type, inter_layer_chunk_size, layer.input_shape, layer.output_shape, layer.attention_type)
            chunk_ready_time_key = (layer.type, pipelining, parallelism, acc_type, inter_layer_chunk_size)


            if latency_key not in latency_cache:
                print(f"Latency estimation for the {acc.type_tag} accelerator type: (chunk_size={inter_layer_chunk_size})")
                result = acc.estimate_cost(layer, pipelining=pipelining, parallelism=parallelism, inter_layer_chunk_size=inter_layer_chunk_size)

                if isinstance(result, tuple):
                    latency, chunk_ready_times = result
                    if layer.type == "atten":
                        if chunk_ready_time_key in chunk_ready_time_cache:
                            # Merge with existing entries
                            existing_times = chunk_ready_time_cache[chunk_ready_time_key]
                            for k, v in chunk_ready_times.items():
                                if v > existing_times[k]:
                                    chunk_ready_time_cache[chunk_ready_time_key][k] = v
                        else:
                            chunk_ready_time_cache[chunk_ready_time_key] = chunk_ready_times
                    else:
                        chunk_ready_time_cache[chunk_ready_time_key] = chunk_ready_times
                else:
                    latency = result
                    chunk_ready_time_cache[chunk_ready_time_key] = {}

                latency_cache[latency_key] = latency
                print(f"{latency:.6f}")

            latencies_by_type[acc.name] = latency_cache[latency_key]
            chunk_ready_times_by_type[acc.name] = chunk_ready_time_cache.get(chunk_ready_time_key, {})

        scored_cands = []
        for acc in cands:
            latency = latencies_by_type[acc.name]
            chunk_ready_times = chunk_ready_times_by_type.get(acc.name, {}) # 
            ready_time = chunk_ready_times.get(chunk_idx, 0.0) if isinstance(chunk_ready_times, dict) and len(chunk_ready_times)>1 else 0.0
            #if ready_time == latency: ready_time = 0
            start_time = find_earliest_non_overlapping_start(acc.schedule, t_ready, ready_time, latency)
            end_time = start_time + latency
            power = acc.estimate_power(layer)
            scored_cands.append((end_time, power, latency, acc, ready_time))

        best_end_time, best_power, latency, best_accel, ready_time = pick_best(scored_cands,digital_opt)
        return best_accel, latency, ready_time
    
    def compute_ready_time(node, scheduled, graph, chunk_ready_time_cache=None, chunk_size=None, link=0):
        """
        Compute when a node's input data is ready, based on predecessors.
        For attention -> FC pipelining, adjusts t_ready using chunked latency.
        """
        import numpy as np

        t_ready = 0.0
        chunk_idx = graph.nodes[node].get("chunk_idx", None)

        for pred in graph.predecessors(node):
            if link:
                m_chunk_pred = re.search(r'_chunk(\d+)', pred)
                m_chunk_node = re.search(r'_chunk(\d+)', node)
                if m_chunk_pred and m_chunk_node:
                    base_name_pred    = pred[:m_chunk_pred.start()]
                    base_name_node    = node[:m_chunk_node.start()]
                    if base_name_pred == base_name_node:
                        continue
            pred_start, pred_dur, _ = scheduled.get(pred, (0.0, 0.0, None))
            pred_end = pred_start + pred_dur
            pred_type = graph.nodes[pred].get("layer", {}).type if "layer" in graph.nodes[pred] else None

            # Special case: attention -> fc chunk
            if (
                chunk_ready_time_cache
                and chunk_idx is not None
                and pred_type == "atten"
            ):
                # Try any matching latency key
                for (ltype, _, _, _, csize), chunk_dict in chunk_ready_time_cache.items():
                    if ltype == "atten" and csize == chunk_size and (not node.startswith("KV_load")):
                        latency = chunk_dict.get(chunk_idx)
                        if latency is not None:
                            pred_end = pred_start + float(latency)
                            break

            t_ready = max(t_ready, pred_end)

        return t_ready

        # --- 3) a compute_ready_time that adds the inter-block barrier ---
    def compute_ready_time_with_block_barrier(node, scheduled):
        # first, normal chunk-and-attention--aware ready-time
        t_ready = compute_ready_time(node, scheduled, chunked_graph, chunk_ready_time_cache, chunk_size)

        # now enforce the "chunk i of block b" -> "chunk i of block b+1" dependency
        # parse out the chunk index and block index from `node`
        #   node names look like "<orig_name>_chunk<i>"
        m = re.match(r"(.+)_chunk(\d+)", node)
        if not m:
            return t_ready

        orig_name = m.group(1)
        chunk_i   = int(m.group(2))

        # find its block index
        mb = re.search(r"_b(\d+)$", orig_name)
        b = int(mb.group(1)) if mb else 0
        if b == 0:
            return t_ready

        # find the _last_ layer of block (b-1), build its chunk name
        prev_last_orig = chunked_graph.nodes[last_of_block[b-1]]['layer'].name
        split_search = re.search(r'(_split\d+)$', prev_last_orig)   # look for _splitN at the end
        if split_search:
            split_tmp = split_search.group(1)             
            layer_name_tmp = prev_last_orig[:split_search.start()]  
        else:
            split_tmp = ""
            layer_name_tmp = prev_last_orig
        
        prev_chunk_node = f"{layer_name_tmp}_chunk{chunk_i}{split_tmp}" #f"{prev_last_orig}_chunk{chunk_i}"

        # if we've already scheduled it, force t_ready >= its finish time
        if prev_chunk_node in scheduled:
            st, du, _ = scheduled[prev_chunk_node]
            t_ready = max(t_ready, st + du)

        return t_ready
    
    def split_info(s):
        # returns (base_name, split_number)
        m = re.match(r'(.*?)(?:_split(\d+))?$', s)
        base = m.group(1)
        num  = int(m.group(2)) if m.group(2) is not None else -1
        return base, num

    # --- 4) now the main scheduling loop, almost identical to schedule_acim_pmca_pipelined_blocks ---
    accel_by_name = {a.name: a for a in accel_instances}
    chip_id = 0
    link_inst = next(a for a in accel_instances if dechip(a.name).startswith("Link[")) #ADDED
    scheduled       = {}
    schedule_dict   = {a.name: [] for a in accel_instances}
    all_slots = [slot for summ in mapping_summary.values() for slot in summ["slots"]]
    num_tiers = max(t for _, t in all_slots) + 1 if all_slots else 1
    # reuse your existing caches
    latency_cache           = {}
    chunk_ready_time_cache  = {}
    transfer_cache = {}
    seen_atten = []

    for node in nx.topological_sort(chunked_graph):
        layer     = chunked_graph.nodes[node]['layer']
        chunk_idx = chunked_graph.nodes[node].get("chunk_idx", None)
        name      = layer.name
        chunk = chunk_size_list[chunk_idx] if chunk_idx else chunk_size_list[0]

        # instead of the old compute_ready_time, we plug in ours:
        t_ready = compute_ready_time_with_block_barrier(node, scheduled)

        m = re.match(r"(atten\d*_b\d+)", name)
        prefix = m.group(1) if m else None
        t_ready_link = t_ready
        # ---------------------- LINK TRANSFERS ----------------------
        if use_link:
            already_explored_atten = [] # False
            if node.startswith("KV_load"):
                t_ready = t_ready_link
                if node not in transfer_cache or t_ready != transfer_cache[node]:
                    transfer_cache[node] = t_ready
            else:
                for pred in chunked_graph.predecessors(node):
                    if pred.startswith("KV_load"):
                        logger.info(f"[LINK] skip splitted layer {pred}->{node} (DMA takes care of it).")
                        t_ready = max(t_ready,t_ready_link)
                        continue
                    layer_pred = chunked_graph.nodes[pred]['layer']
                    name_pred = layer_pred.name
                    if name_pred == name:
                        t_ready_link = compute_ready_time(
                            node,
                            scheduled,
                            chunked_graph,
                            chunk_ready_time_cache,
                            chunk,
                            link=1
                        )
                        continue
                    m_chunk = re.search(r'_chunk(\d+)', node) 
                    if m_chunk:
                        chunk_suffix = m_chunk.group(0)
                        chunk_id     = m_chunk.group(1)
                        base_name    = name[:m_chunk.start()]
                    else:
                        chunk_suffix = None
                        chunk_id     = None
                        base_name    = name

                    # Skip redundant moves when pred & node share inputs
                    if analog_pool and hasattr(analog_pool, "share_map"):
                        sm = analog_pool.share_map
                        # if preceed by attention only take one of them
                        if pred.startswith("atten") or pred.startswith("gelu") or pred.startswith("layernorm") or pred.startswith("gemm"):
                            if pred in already_explored_atten:
                                continue
                            else:
                                already_explored_atten.append(pred)

                        # do they share?
                        if name in sm:
                            skip_pred = False
                            existing_node = None
                            # both must be analog-placed
                            chunk_suffix = chunk_suffix if chunk_suffix else ""
                            for shared_node in sm[name]:
                                #if name_pred in sm:
                                if shared_node+chunk_suffix in transfer_cache: #and mapping_summary[name]["tiles_used"]==mapping_summary[shared_node]["tiles_used"]:
                                    if name.startswith("residual") and pred.startswith("fc"):
                                        continue
                                    logger.info(f"[LINK] skip shared-input {pred}->{name}.")
                                    skip_pred = True
                                    existing_node = shared_node+chunk_suffix
                                    break
                            if skip_pred:
                                t_ready = max(transfer_cache[existing_node],t_ready_link)
                                continue
                    if layer.type == "residual":
                        continue
                    if prefix and prefix in seen_atten:
                        t_ready = transfer_cache[prefix]
                        continue

                    # when did producer finish?
                    if layer_pred.type == "atten":
                        prod_finish = t_ready_link
                    else :
                        st_pred, du_pred, _ = scheduled[pred]
                        prod_finish = st_pred + du_pred
                    

                    # ask Link how long to move pred's output
                    # (CostModel.link_latencycost reads layer.output_shape)
                    if "split" in node and not "split0" in node:
                        prefix = node.split('_split')[0]
                        if node.startswith("atten"):
                            match_suff = re.search(r"(_chunk\d+)$", node)
                            suffix = match_suff.group(1) if match_suff else ""    
                            logger.info(f"[LINK] skip splitted layer {pred}->{node}.")
                            t_ready = transfer_cache[prefix+"_split0"+suffix]
                            break
                        logger.info(f"[LINK] skip splitted layer {pred}->{node}.")
                        t_ready = transfer_cache[prefix+"_split0"]
                        break
                    elif pred.startswith("atten") == False and "split" in pred :#and not "split0" in pred:
                        if (pred.startswith("gelu")==True and "_r" in pred) and not "split0" in pred:
                            prefix = pred.split('_split')[0]    
                            logger.info(f"[LINK] skip splitted layer {pred}->{node}.")
                            t_ready = transfer_cache[node]
                            break
                        else:
                            xfer_dur = link_inst.estimate_cost(chunked_graph.nodes[pred]['layer'], SL_scale=chunked_graph.nodes[pred]['layer'].input_shape[0]) # chunk
                    else:
                        if node.startswith("atten") and (not pred.startswith("KV_load")):
                            if pred.startswith("fc") and chunked_graph.nodes[pred]['layer'].attention_type == "cross":
                                if prefill <= 1:
                                    xfer_dur = link_inst.estimate_cost(chunked_graph.nodes[pred]['layer'],SL_scale=SL_CROSS)
                                else:
                                    xfer_dur = 0.0
                            else:
                                xfer_dur = link_inst.estimate_cost(chunked_graph.nodes[pred]['layer'],SL_scale=SL)
                        else:
                            if pred.startswith("KV_load"):
                                xfer_dur = 0.0
                            else:
                                xfer_dur = link_inst.estimate_cost(chunked_graph.nodes[pred]['layer'], SL_scale=chunk)

                    busy = sorted([(st, st+dur) for (_, st, dur) in link_inst.schedule], key=lambda x: x[0])
                    t = prod_finish
                    for bs, be in busy:
                        if t + xfer_dur <= bs:
                            break
                        t = max(t, be)
                    start = t
                    end = start + xfer_dur
                    # optional: record it
                    if node.startswith("atten"):
                        #p = re.search(r'_chunk(\d+)$', pred) ###TODO in schedule_acim_pmca_pipelined_blocks
                        p = re.search(r'_chunk(\d+)(?:_split\d+)?$', pred)
                        pred_prefix= pred[:p.start()]
                        link_inst.schedule.append((f"{pred_prefix}->{node}", start, xfer_dur))
                        schedule_dict[link_inst.name].append((f"{pred_prefix if pred_prefix else pred}->{node}", start, xfer_dur))
                        logger.info(f"[LINK] transfer {pred_prefix if pred_prefix else pred}->{node} on {link_inst.name} @ {start:.6f} ms, dur={xfer_dur:.6f} ms")
                    else:
                        link_inst.schedule.append((f"{pred}->{node}", start, xfer_dur))
                        schedule_dict[link_inst.name].append((f"{pred}->{node}", start, xfer_dur))
                        logger.info(f"[LINK] transfer {pred}->{node} on {link_inst.name} @ {start:.6f} ms, dur={xfer_dur:.6f} ms")
                    # bump t_ready so compute only starts after the last transfer
                    t_ready = max(t_ready,end)#, xfer_start + xfer_dur)
                    if node not in transfer_cache or t_ready != transfer_cache[node]:
                        transfer_cache[node] = t_ready

        if prefix and use_link and prefix not in seen_atten:
            seen_atten.append(prefix)
            transfer_cache[prefix] = t_ready

        # -------- Intra-layer split handling (already preprocessed) --------
        if hasattr(layer, "intra_layer_splits"):
            cur_time = t_ready
            for sub in layer.intra_layer_splits:
                best, dur, ready_time = pick_digital_accelerator(sub, cur_time, accel_instances, 
                                                                 pipelining=pipelining, 
                                                                 parallelism=parallelism,
                                                                 latency_cache=latency_cache,
                                                                 chunk_ready_time_cache=chunk_ready_time_cache,
                                                                 model_name=model_name,
                                                                 inter_layer_chunk_size=chunk,
                                                                 chunk_idx=chunk_idx,
                                                                 digital_opt=digital_opt
                                                                 )   
                start = find_earliest_non_overlapping_start(best.schedule, cur_time, cur_time, dur)
                best.available_time = start + dur
                best.schedule.append((sub.name, start, dur))
                schedule_dict[best.name].append((sub.name, start, dur))
                cur_time = start + dur
                logger.info(f"Scheduled layer split {sub.name:6s} on {best.name:20s} @ {start:.3f} dur={dur:.6f}")
            
            scheduled[node] = (t_ready, cur_time - t_ready, best.name)
            logger.info(f"Scheduled {name:6s} on {best.name:20s} @ {t_ready:.3f} dur={cur_time - t_ready:.6f}")
            continue

        if layer.type in ("fc", "relu", "batch_norm"):
            slots = mapping_summary[name]['slots']
            base_dur = mapping_summary[name]['latency']
            col_blocks = mapping_summary[name]['col_blocks']

            if layer.attention_type == "cross" and prefill <= 1:
                dur = scale_analog_latency(base_dur, SL_CROSS)
            elif layer.attention_type == "cross" and prefill > 1:
                dur = 0.0
            else:
                dur = scale_analog_latency(base_dur, chunk)

            start = t_ready
            while True:
                new_start, bumped = check_analog_conflicts(start, dur, slots, col_blocks, mapping_summary, accel_by_name, num_tiers, analog_pool.share_map, layer.name, chip_id)
                if bumped:
                    start = new_start
                else:
                    break
            acc_names = []
            for tile_id, tier_id in slots:
                inst_name = chipify(f"ACIMTile[{tile_id},{tier_id}]", chip_id)
                inst = accel_by_name[inst_name]
                inst.schedule.append((name, start, dur))
                schedule_dict[inst_name].append((node, start, dur))
                acc_names.append(inst_name)
            scheduled[node] = (start, dur, ",".join(acc_names))

        elif layer.type == "residual":
            #print(f"[PPU Residual] Processing {name}")
            logger.info(f"[PPU Residual] Processing {name}")
            src_layers = list(chunked_graph.predecessors(node))
            src_layers_pred_chunk = list(chunked_graph.predecessors(node))
            src_layers.sort(key=lambda n: scheduled[n][0], reverse=True)
            not_from_analog = False
            src_layers = [re.sub(r"_chunk\d+$", "", src) for src in src_layers]            # Find analog-mapped tiles from predecessors
            
            # 2) record base-order as they appear
            base_order = []
            for s in src_layers:
                b, _ = split_info(s)
                if b not in base_order:
                    base_order.append(b)
            base_index = {b:i for i,b in enumerate(base_order)}

            # 4) final sort key: by base's original index, then by split number
            src_layers = sorted(
                src_layers,
                key=lambda s: (
                    base_index[split_info(s)[0]],
                    split_info(s)[1]
                )
            )

            mapped_tiles = []
            for src in src_layers:
                if src in mapping_summary:
                    mapped_tiles.extend(mapping_summary[src]["slots"])

            if not mapped_tiles:
                logger.info(f"[PPU Residual] no ACIM predecessor for {name}. TMP")
                not_from_analog = True
                for analog_layer in mapping_summary:
                    if analog_layer.startswith("fc"):
                        mapped_tiles.extend(mapping_summary[analog_layer]["slots"])

            m_chunk = re.search(r'_chunk(\d+)$', node)
            chunk_suffix = m_chunk.group(0)
            # Try scheduling on each mapped tile
            assigned = False
            for tile_id, tier_id in mapped_tiles:
                inst_name = chipify(f"ACIMTile[{tile_id},{tier_id}]", chip_id)
                inst = accel_by_name[inst_name]

                # ---- 1) LINK transfers for preds NOT on this (tile_id,tier_id) ----
                if use_link:
                    for src in src_layers:
                        # if this src *is* on our candidate tile, skip the link hop
                        if (tile_id, tier_id) in mapping_summary.get(src, {}).get("slots", []):
                            logger.info(f"[LINK] skip ext-tile-acc {src}->{name} on {inst_name}")
                            continue
                        if (layer.name in sm and not src.startswith("fc")) or src.startswith("residual"):
                            continue                        
                        chunk_suffix_pred = [s for s in src_layers_pred_chunk if s.startswith(src)]
                        # otherwise, schedule a LINK transfer
                        st_pred, du_pred, _ = scheduled[chunk_suffix_pred[0]]
                        prod_finish = st_pred + du_pred

                        if src.startswith("atten") == False and "split" in src :#and not "split0" in pred:
                            if (src.startswith("gelu")==True and "_r" in src) and not "split0" in src:
                                prefix = src.split('_split')[0]    
                                logger.info(f"[LINK] skip splitted layer {src}->{node}.")
                                t_ready = transfer_cache[node]
                                break
                            else:
                                xfer_dur = link_inst.estimate_cost(chunked_graph.nodes[chunk_suffix_pred[0]]['layer'], SL_scale=chunked_graph.nodes[chunk_suffix_pred[0]]['layer'].input_shape[0]) # chunk
                        else:
                            xfer_dur   = link_inst.estimate_cost(chunked_graph.nodes[chunk_suffix_pred[0]]['layer'], SL_scale=chunk)

                        busy = sorted([(st, st+dur) for (_, st, dur) in link_inst.schedule], key=lambda x: x[0])
                        t = prod_finish
                        for bs, be in busy:
                            if t + xfer_dur <= bs:
                                break
                            t = max(t, be)
                        start = t
                        end = start + xfer_dur
                        link_inst.schedule.append((f"{chunk_suffix_pred[0]}->{node}", start, xfer_dur))
                        schedule_dict[link_inst.name].append((f"{chunk_suffix_pred[0]}->{node}", start, xfer_dur))
                        logger.info(f"[LINK] transfer {chunk_suffix_pred[0]}->{node} on {link_inst.name} "
                                    f"@ {start:.6f} ms, dur={xfer_dur:.6f} ms")
                        t_ready = max(t_ready,end)
                        transfer_cache[node] = t_ready

                # Check if this tile was also used by any predecessor -> enables ext_tile_acc
                ext_tile_acc = any(
                    (tile_id, tier_id) in mapping_summary.get(src, {}).get("slots", [])
                    for src in src_layers
                )

                # Estimate latency using GeneralAnalogTilePool API
                latency = analog_pool.estimate_latency_residual(
                    layer_size=layer.input_shape[0], SL=chunk, ext_tile_acc=ext_tile_acc, not_from_analog=not_from_analog
                )

                start = t_ready
                while True:
                    res_conflict_end = min(
                        (s + d for lname, s, d in inst.schedule
                        if lname.startswith("residual")
                        and not (s + d <= start or start + latency <= s)),
                        default=None
                    )
                    if not_from_analog:
                        fc_conflict_end = min(
                            (s + d for lname, s, d in inst.schedule
                            if lname.startswith("fc")
                            and not (s + d <= start or start + latency <= s)),
                            default=None
                        )
                    else:
                        fc_conflict_end = None
                    res_conflict = res_conflict_end is not None
                    fc_conflict = fc_conflict_end is not None
                    if res_conflict or fc_conflict:
                        start = max(e for e in (res_conflict_end, fc_conflict_end) if e is not None) #0.0001  # step forward slightly to resolve conflict
                    else:
                        break

                # Schedule residual on PPU
                inst.schedule.append((name, start, latency))
                schedule_dict[inst_name].append((node, start, latency))
                scheduled[node] = (start, latency, inst_name)

                logger.info(f"Scheduled residual {name} on {inst_name} @ {start:.6f} dur={latency:.6f} ms (ext_tile_acc={ext_tile_acc}, not_from_analog={not_from_analog})")
                assigned = True
                break

            if not assigned:
                layer.input_shape = (chunk_size, layer.input_shape[0]) #we are moving the mapping of residual to DAs/PMCAs which need the squared size
                best, dur, ready_time = pick_digital_accelerator(
                    layer, t_ready, accel_instances,
                    pipelining=pipelining, parallelism=parallelism,
                    latency_cache=latency_cache,
                    chunk_ready_time_cache=chunk_ready_time_cache,
                    model_name=model_name,
                    inter_layer_chunk_size=chunk,
                    chunk_idx=chunk_idx,
                    digital_opt=digital_opt
                )
                start = find_earliest_non_overlapping_start(best.schedule, t_ready, ready_time or 0.0, dur)
                best.available_time = start + dur
                best.schedule.append((name, start, dur))
                schedule_dict[best.name].append((node, start, dur))
                scheduled[node] = (start, dur, best.name)

            #continue
        else:
            best, dur, ready_time = pick_digital_accelerator(
                layer, t_ready, accel_instances,
                pipelining=pipelining, parallelism=parallelism,
                latency_cache=latency_cache,
                chunk_ready_time_cache=chunk_ready_time_cache,
                model_name=model_name,
                inter_layer_chunk_size=chunk,
                chunk_idx=chunk_idx,
                digital_opt=digital_opt
            )
            start = find_earliest_non_overlapping_start(best.schedule, t_ready, ready_time or 0.0, dur)
            best.available_time = start + dur
            best.schedule.append((name, start, dur))
            schedule_dict[best.name].append((node, start, dur))
            scheduled[node] = (start, dur, best.name)

        st, du, an = scheduled[node]
        chunked_graph.nodes[node].update({"accelerator": an, "start_time": st, "duration": du})
        logger.info(f"Scheduled {node:6s} on {an:20s} @ {st:.6f} dur={du:.6f}")
    
    # at the end, collect final latency
    starts = [st for st,_,_ in scheduled.values()]
    ends   = [st+du for st,du,_ in scheduled.values()]
    total_lat = max(ends) - min(starts)
    logger.info(f"[Block level pipeline Schedule Complete ({chunk_size_list[0]})] total latency: {total_lat:.6f} ms")

    return schedule_dict, total_lat

