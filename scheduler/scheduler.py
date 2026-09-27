#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#

from AnalogMapping import GeneralAnalogTilePool
from scheduler.analog_scheduler import premap_analog_layers
from nodes.cost_models import CostModel, PowerModel, MemoryEstimator
from nodes.nodes_definition import AcceleratorInstance
from nodes.accelerator_config import *
from utils.memory_split import LayerSplitter
from utils.layer_splitter_utils import split_layer_intra, get_min_tcdm_capacity_for_op
from utils.chip_utils import chipify
import math
from models.model_trial import Layer
import networkx as nx

class Scheduler:
    """
    Orchestrates the scheduling and mapping of layers onto a hybrid architecture.
    Applies analog pre-mapping and digital scheduling with configurable parallelism,
    pipelining, and optimization goals (latency, power, area).
    """
    def __init__(self, optimization_level="latency", parallelism_type="none", pipelining_type="none", 
                 num_tiles=16, num_tiers=1, num_pmca_red=None, num_pmca=None, num_da0=None, share_map=None, 
                 model_name="None", SL=1, chip_id: int = 0,
                 flash_attention=False, sl_warp=None, auto_tune_sl_warp = False, autoregressive=False, prefill_size = None, SL_CROSS=None,
                 logger=None, digital_opt="latency"):
        self.optimization_level = optimization_level
        self.parallelism_type = parallelism_type
        self.pipelining_type = pipelining_type
        self.num_tiles = num_tiles
        self.num_tiers = num_tiers
        self.share_map = share_map
        self.model_name = model_name
        self.SL = SL
        self.logger = logger
        self.flash_attention = flash_attention
        self.auto_tune_sl_warp = auto_tune_sl_warp
        self.sl_warp = sl_warp
        self.autoregressive = autoregressive
        self.prefill_size = prefill_size
        self.SL_CROSS = SL_CROSS
        self.chip_id = chip_id
        self.digital_opt = digital_opt

        # Flags for cost model
        self.intra_layer_parallelism = False
        self.intra_layer_pipelining = False
        self.inter_layer_parallelism = False
        self.inter_layer_pipelining = False
        self.optimize_attention_chunk = False

        # Set num_pmca_red and num_pmca depending on optimization level
        self.set_digital_resources(num_pmca_red, num_pmca, num_da0)

        # Initialize analog pool
        self.analog_pool = GeneralAnalogTilePool(
            num_tiles=self.num_tiles,
            optimize_for=self.set_analog_optimization(),
            share_map=self.share_map,
            chip_id=self.chip_id
        )

    def set_digital_resources(self, num_pmca_red, num_pmca, num_da0):
        """
        Set number of PMCA_RED and PMCA instances depending on optimization goal.
        """
        total_pmca_red = PMCA_RED_TILE_CONFIG["num_tiles"]
        total_pmca = PMCA_TILE_CONFIG["num_tiles"]
        total_da0 = DA0_TILE_CONFIG["num_tiles"]

        if self.optimization_level in ["area", "power"]:
            self.num_pmca_red = 1 if total_pmca_red > 0 else 0
            self.num_pmca = 1 if total_pmca > 0 else 0
            self.num_da0 = 1 if total_da0 > 0 else 0
        elif self.optimization_level == "latency":
            self.num_pmca_red = num_pmca_red if num_pmca_red is not None else total_pmca_red
            self.num_pmca = num_pmca if num_pmca is not None else total_pmca
            self.num_da0 = num_da0 if num_da0 is not None else total_da0
        elif self.optimization_level == "area_latency_balance":
            self.num_pmca_red = math.ceil(total_pmca_red / 2)
            self.num_pmca = math.ceil(total_pmca / 2)
            self.num_da0 = math.ceil(total_da0 / 2)
        else:
            raise ValueError(f"Unsupported optimization level: {self.optimization_level}")

        log = self.logger.info if self.logger else print
        log(f"[Scheduler] Using {self.num_pmca_red} PMCA_RED and {self.num_pmca} PMCA and {self.num_da0} DA0")

    def set_analog_optimization(self):
        """
        Set the analog optimization level based on the chosen optimization level.
        """
        if self.optimization_level == "latency":
            return 'latency'
        elif self.optimization_level in ["area", "power"]:
            return 'area'
        elif self.optimization_level == "area_latency_balance":
            return 'area_latency_balance'
        else:
            raise ValueError(f"Unknown optimization level: {self.optimization_level}")

    def analog_mapping(self, graph, display_plots=True):
        """
        Perform the analog pre-mapping of layers and handle optimization based on the selected option.
        """
        # Perform analog mapping, depending on the optimization level
        premap_analog_layers(graph, self.analog_pool)
        # Debug print and plot to verify the mapping process
        self.analog_pool.debug_print(model_name=self.model_name)
        if display_plots:
            self.analog_pool.plot(model_name=self.model_name, rows=int(math.sqrt(self.num_tiles)))

        # Return the updated mapping details for analog layers
        mapping_summary, max_tile_used_single_tier, max_tile_used = self.analog_pool.summarize_mapping_inferred(model_name=self.model_name)
        return mapping_summary, max_tile_used_single_tier, max_tile_used

    def optimize_mapping(self, graph, accelerators):
        """
        Perform the analog-side optimization for mapping layers based on optimization level.
        This will adjust the analog mapping and report the results.
        """
        # --- Check memory requirements for pre-fill/autoregressive steps ---
        self.check_onchip_SRAM_requirements(graph, accelerators, autoregressive=self.autoregressive, prefill_size=self.prefill_size)
        # --- Parallelism Handling ---
        if self.parallelism_type in ("intra_layer", "both"):
            self.apply_intra_layer_parallelism(graph, accelerators)
        if self.parallelism_type in ("inter_layer", "both"):
            self.apply_inter_layer_parallelism(graph, accelerators)

        #Check memory constraints for all digital layers
        if self.pipelining_type not in ("inter_intra_layer", "inter_block", "inter_layer"):
            self.check_and_adjust_layer_memory(graph, accelerators)

        # --- Pipelining Handling ---
        if self.pipelining_type in ("intra_layer"):
            self.apply_intra_layer_pipelining(graph, accelerators)
            if self.autoregressive:
                self.optimize_attention_chunk = False 
            else:
                self.optimize_attention_chunk = True 
        if self.pipelining_type in ("inter_layer"):
            self.apply_inter_layer_pipelining(graph, accelerators)
            self.optimize_attention_chunk = False 
        if self.pipelining_type in ("inter_intra_layer", "inter_block"):
            self.optimize_attention_chunk = False 
            self.apply_intra_layer_pipelining(graph, accelerators)
            self.apply_inter_layer_pipelining(graph, accelerators)

    def check_and_adjust_layer_memory(self, graph, accelerators):
        """
        For each digital layer or sublayer, check if it fits in TCDM.
        If not, split it recursively to make it fit.
        """
        min_tcdm_capacity_att = get_min_tcdm_capacity_for_op("atten", accelerators)
        min_tcdm_capacity_layernorm = get_min_tcdm_capacity_for_op("layernorm", accelerators)
        min_tcdm_capacity_gelu = get_min_tcdm_capacity_for_op("gelu", accelerators)
        for node in list(graph.nodes()):
            layer = graph.nodes[node]['layer']
            if layer.type not in ("atten", "layernorm", "gelu"):
                continue  # Only check digital layers

            if layer.type in ("layernorm", "gelu") and self.pipelining_type in ("inter_layer","inter_intra_layer","inter_block"):
                continue # Skip if inter-layer pipelining is enabled

            # Operate on pre-split sublayers if they exist
            sublayers = getattr(layer, "parallel_sublayers", [layer])
            for sub in sublayers:
                mem_kb = MemoryEstimator.estimate_memory_kb(sub, flash_attention=self.flash_attention)
                if sub.type == "atten":
                    if mem_kb > min_tcdm_capacity_att:
                        sub.force_serial = True
                        sub.intra_layer_splits = LayerSplitter.split_attention_to_fit_tcdm(sub, min_tcdm_capacity_att)
                elif sub.type == "layernorm":
                    if mem_kb > min_tcdm_capacity_layernorm:
                        sub.force_serial = True
                        sub.intra_layer_splits = LayerSplitter.split_sublayer_to_fit_tcdm(sub, min_tcdm_capacity_layernorm)
                elif sub.type == "gelu":
                    if mem_kb > min_tcdm_capacity_gelu:
                        sub.force_serial = True
                        sub.intra_layer_splits = LayerSplitter.split_sublayer_to_fit_tcdm(sub, min_tcdm_capacity_gelu)

    def create_accelerator_instances(self, mapping_summary=None):
        """
        Create instances of accelerators (PMCA and ACIM tiles) dynamically.
        """
        accelerator_instances = []
        chip = self.chip_id  
        # Create PMCA instances
        for i in range(self.num_pmca_red):
            inst_name = chipify(f"PMCA_RED[{i}]", chip)
            accelerator_instances.append(AcceleratorInstance(
                inst_name, 
                "PMCA_RED", 
                ["atten", "layernorm", "gelu", "softmax", "gemm", "residual"], 
                lambda layer, pipelining=None, parallelism=None, inter_layer_chunk_size=None: 
                CostModel.PMCA_cost(layer, 
                                    intra_layer_parallelism=self.intra_layer_parallelism, 
                                    intra_layer_pipelining=self.intra_layer_pipelining,
                                    inter_layer_pipelining=self.inter_layer_pipelining, 
                                    inter_layer_chunk_size=inter_layer_chunk_size,
                                    gemm_fast=True, 
                                    optimize_attention_chunk=self.optimize_attention_chunk,
                                    flash_attention=self.flash_attention,
                                    auto_tune_sl_warp=self.auto_tune_sl_warp,
                                    sl_warp=self.sl_warp,
                                    prefill_size=self.prefill_size,
                                    autoregressive=self.autoregressive,
                                    model_name=self.model_name,
                                    sl_cross = self.SL_CROSS),
                PowerModel.PMCA_RED_powercost,
                chip_id=chip
            ))

        for i in range(self.num_pmca):
            inst_name = chipify(f"PMCA[{i}]", chip)
            accelerator_instances.append(AcceleratorInstance(
                inst_name, 
                "PMCA", 
                ["atten", "layernorm", "gelu", "softmax", "gemm", "residual"], 
                lambda layer, pipelining=None, parallelism=None, inter_layer_chunk_size=None: 
                CostModel.PMCA_cost(layer, 
                                    intra_layer_parallelism=self.intra_layer_parallelism, 
                                    intra_layer_pipelining=self.intra_layer_pipelining, 
                                    inter_layer_pipelining=self.inter_layer_pipelining,
                                    inter_layer_chunk_size=inter_layer_chunk_size,
                                    gemm_fast=False, 
                                    optimize_attention_chunk=self.optimize_attention_chunk,
                                    flash_attention=self.flash_attention,
                                    auto_tune_sl_warp=self.auto_tune_sl_warp,
                                    sl_warp=self.sl_warp,
                                    prefill_size=self.prefill_size,
                                    autoregressive=self.autoregressive,
                                    model_name=self.model_name,
                                    sl_cross = self.SL_CROSS),
                PowerModel.PMCA_powercost,
                chip_id=chip
            ))

        # Create ACIM tile instances
        for i in range(self.num_tiles):
            for j in range(self.num_tiers):
                inst_name = chipify(f"ACIMTile[{i},{j}]", chip)
                accelerator_instances.append(AcceleratorInstance(
                        inst_name, "ACIMTile", ["fc", "relu", "batch_norm", "residual"],
                        lambda layer, i=i, j=j: self.analog_pool.estimate_latency(layer.name)[0],
                        lambda layer, i=i, j=j: mapping_summary[layer]["estimated_power"],
                        chip_id=chip
                ))

        # Create DA0 tile instances
        for i in range(self.num_da0):
            inst_name = chipify(f"DA0[{i}]", chip)
            accelerator_instances.append(AcceleratorInstance(
                inst_name, 
                "DA0", 
                ["layernorm", "gelu", "atten", "softmax", "gemm"], 
                lambda layer, pipelining=None, parallelism=None, inter_layer_chunk_size=None: 
                CostModel.DA_cost(layer, 
                                    da_type="DA0",
                                    intra_layer_parallelism=self.intra_layer_parallelism, 
                                    intra_layer_pipelining=self.intra_layer_pipelining,
                                    inter_layer_pipelining=self.inter_layer_pipelining, 
                                    inter_layer_chunk_size=inter_layer_chunk_size,
                                    optimize_attention_chunk=self.optimize_attention_chunk,
                                    flash_attention=self.flash_attention,
                                    model_name=self.model_name),
                PowerModel.DA0_powercost,
                chip_id=chip
            ))
                
        # --- New: SRAM tiles for low-latency fetches (e.g. small hot buffers) ---
        for i in range(SRAM_TILE_CONFIG["num_tiles"]):
            inst_name = chipify(f"SRAM[{i}]", chip)
            accelerator_instances.append(AcceleratorInstance(
                inst_name,
                "SRAM",
                ["storage"],
                # cost is purely a latency to fetch N bytes from SRAM
                lambda layer, pipelining=None, parallelism=None, inter_layer_chunk_size=None:
                    CostModel.SRAM_latency(layer.num_bytes),
                # power model for SRAM read
                lambda layer: PowerModel.SRAM_powercost(layer.num_bytes)
            ))

        # --- DDR for bulk DRAM transfers (e.g. initial embedding load) ---
        for i in range(DDR_TILE_CONFIG["num_tiles"]):
            inst_name = chipify(f"DDR[{i}]", chip)
            accelerator_instances.append(AcceleratorInstance(
                inst_name,
                "DDR",
                # supports any fetch that doesn't fit in SRAM
                ["embedding", "KV_load"],
                lambda layer, pipelining=None, parallelism=None, inter_layer_chunk_size=None:
                    CostModel.DDR_latencycost(layer),
                lambda layer: PowerModel.DDR_powercost(layer),
                chip_id=chip
            ))

        # --- New: single-port 64 b/cycle interconnect "Link" ---
        for i in range(LINK_CONFIG["num_links"]):
            inst_name = chipify(f"Link[{i}]", chip)
            accelerator_instances.append(AcceleratorInstance(
                inst_name,
                "Link",
                # We'll say *every* layer's output must traverse this link
                # before the next layer can start.  You could restrict types.
                ["fc","relu","batch_norm","residual","atten","gelu","layernorm","softmax"],
                # cost = time to push layer.output_shape bits across 64 b/cycle
                lambda layer, pipelining=None, parallelism=None, inter_layer_chunk_size=None, SL_scale=1.0:
                    CostModel.link_latencycost(layer, SL_scale=SL_scale),
                lambda layer: PowerModel.Link_powercost(layer),
                chip_id=chip
            ))

        return accelerator_instances

    def apply_inter_layer_parallelism(self, graph, accelerators):
        """
        Schedule independent layers concurrently on different accelerators.
        """
        self.inter_layer_parallelism = True

    def apply_intra_layer_parallelism(self, graph, accelerators):
        """
        Apply intra-layer parallelism by splitting attention layers across multiple PMCAs.
        """
        self.intra_layer_parallelism = True

        nodes_to_add, nodes_to_remove = [], []

        for node in list(graph.nodes()):
            layer = graph.nodes[node]['layer']
            if layer.type in ("atten") or (layer.type in ("layernorm", "gelu", "gemm") and (self.pipelining_type not in ("inter_layer","inter_intra_layer","inter_block"))):#, "layernorm", "gelu"):
                # Call the new general split function
                split_layers = split_layer_intra(layer, accelerators, self.digital_opt)

                if len(split_layers) > 1:
                    nodes_to_remove.append(node)

                    for idx, sublayer in enumerate(split_layers):
                        new_node = f"{node}_split{idx}"
                        nodes_to_add.append((new_node, sublayer, node))

        # --- Actually modify the graph ---
        for node in nodes_to_remove:
            preds = list(graph.predecessors(node))
            succs = list(graph.successors(node))
            graph.remove_node(node)

            for new_node, sublayer, original_node in nodes_to_add:
                if original_node == node:
                    graph.add_node(new_node, layer=sublayer)

                    for pred in preds:
                        graph.add_edge(pred, new_node)

                    for succ in succs:
                        graph.add_edge(new_node, succ)

    def apply_inter_layer_pipelining(self, graph, accelerators):
        """
        Begin execution of layer N+1 while N is still running, if dependencies permit.
        """
        self.inter_layer_pipelining = True

    def apply_intra_layer_pipelining(self, graph, accelerators):
        """
        Overlap execution of different sub-ops within a single layer (e.g., attention chunks).
        """
        self.intra_layer_pipelining = True

    def check_onchip_SRAM_requirements(self, graph, accelerators, autoregressive=False, prefill_size=None):
        """
        Check if on-chip SRAM can accommodate pre-fill/autoregressive buffers.
        Raise error if not enough memory is available for at least prefill.
        """
        sram_tile_bytes = SRAM_TILE_CONFIG['size'] * 2**20  # B per tile
        total_sram_bytes = sram_tile_bytes * SRAM_TILE_CONFIG['num_tiles']
        tot_bytes_per_model = 0   
        bytes_per_block = 0

        for node in list(graph.nodes()):
            layer = graph.nodes[node]['layer']
            if layer.type == "atten" and autoregressive:
                if prefill_size is None:
                    RuntimeError("Prefill size must be specified for autoregressive models.")
                # Estimate memory for pre-fill and autoregressive buffers
                if layer.attention_type == "cross":
                    # For cross-attention, we need to consider the SL_CROSS for K and V instead of the main SL
                    sequence_length = layer.input_shape[0]
                    d_model = layer.output_shape[1] 
                    Q_plus_out = sequence_length * d_model * 2 * ACIM_TILE_CONFIG['precision'] / 8  # bytes for Q and output concatenated
                    K_plus_V = self.SL_CROSS * d_model * 2  * ACIM_TILE_CONFIG['precision'] / 8   # bytes for K and V concatenated
                    bytes_per_block =  Q_plus_out + K_plus_V  # bytes
                    tot_bytes_per_model += bytes_per_block  # accumulate for all attention layers
                else:
                    sequence_length = layer.input_shape[0]
                    d_model = layer.output_shape[1] 
                    Q_plus_out = sequence_length * d_model * 2 * ACIM_TILE_CONFIG['precision'] / 8  # bytes for Q and output concatenated
                    K_plus_V = prefill_size * d_model * 2  * ACIM_TILE_CONFIG['precision'] / 8   # bytes for K and V concatenated
                    bytes_per_block =  Q_plus_out + K_plus_V  # bytes
                    tot_bytes_per_model += bytes_per_block  # accumulate for all attention layers
            elif layer.type == "atten" and not autoregressive:
                # Estimate memory for pre-fill buffers only
                sequence_length = layer.input_shape[0]
                d_model = layer.output_shape[1]  # 
                Q_plus_out = sequence_length * d_model * 2 * ACIM_TILE_CONFIG['precision'] / 8 # bytes for Q and output concatenated
                K_plus_V = sequence_length * d_model * 2  * ACIM_TILE_CONFIG['precision'] / 8   # bytes for K and V concatenated
                bytes_per_block =  Q_plus_out + K_plus_V   # bytes
                tot_bytes_per_model = bytes_per_block  # total memory requirement is just for a single layer, we do not need to store them for next timestep

        if bytes_per_block:
            blocks_fit_total = total_sram_bytes // bytes_per_block
            blocks_fit_per_sram = sram_tile_bytes // bytes_per_block

            if blocks_fit_total<1:
                raise MemoryError(f"Insufficient on-chip SRAM: Required {bytes_per_block/1024} KB (1 block), Available {total_sram_bytes/1024} KB")
            
            if autoregressive and tot_bytes_per_model > total_sram_bytes:
                # overlap == True -> we can ping-pong (KV_load and atten can overlap)
                overlap = (blocks_fit_per_sram == 1 and SRAM_TILE_CONFIG["num_tiles"] >=2) or (blocks_fit_per_sram >=2)
                self.insert_kv_reload_ops(graph, overlap=overlap, blocks_fit_total=blocks_fit_total, KV_size=math.ceil(K_plus_V))

    def insert_kv_reload_ops(self, graph, overlap: bool, blocks_fit_total: int, KV_size: int):
        """
        Insert KV_load nodes in the graph for each attention layer.

        If overlap == False:
            serialize KV_load and attention (no ping-pong).
        If overlap == True:
            allow KV_load_i to start as soon as its data dependencies are ready,
            so it can overlap with attention of previous blocks/ layers.
        """
          # adjust import if needed

        # Get attention nodes in topological order
        topo = list(nx.topological_sort(graph))
        atten_nodes = [n for n in topo if graph.nodes[n]['layer'].type == "atten"]

        prev_atten = []
        for idx, att_node in enumerate(atten_nodes):
            if idx < blocks_fit_total:
                prev_atten.append(att_node)
                continue

            att_layer = graph.nodes[att_node]['layer']
            seq_len   = att_layer.input_shape[0]

            kv_name = f"KV_load_{att_node}"
            kv_layer = Layer(kv_name, "KV_load", (seq_len,), (KV_size,))

            # Add node with same accelerator-type metadata you use elsewhere
            graph.add_node(
                kv_name,
                layer=kv_layer
            )

            # Serialize or overlap with previous attention
            if len(prev_atten) >= 1:
                if not overlap and prev_atten is not None:
                    # Force KV_load_i to wait for atten_{i-1}
                    graph.add_edge(prev_atten.pop(0), kv_name)
                else:
                    atten_dependence = prev_atten.pop(0)
                    graph.add_edge(atten_dependence, kv_name)
            
            # 3) Finally, KV_load -> atten
            graph.add_edge(kv_name, att_node)

            prev_atten.append(att_node)
    