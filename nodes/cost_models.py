#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#

from nodes.accelerator_config import PMCA_RED_TILE_CONFIG, PMCA_TILE_CONFIG, DA0_TILE_CONFIG, DDR_TILE_CONFIG, ACIM_TILE_CONFIG, LINK_CONFIG, SRAM_TILE_CONFIG
from pmca_prediction_model.load_pred_model import PMCA_pred_models
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import re
from datetime import datetime
from pathlib import Path

# Global latency cache for attention chunk evaluation
global_chunk_latency_cache = {}
global_warp_latency_cache = {}

class CostModel:
    @staticmethod
    def PMCA_cost(layer,
                  intra_layer_parallelism=False,
                  intra_layer_pipelining=False,
                  inter_layer_pipelining=False,
                  gemm_fast=False,
                  optimize_attention_chunk=True,
                  flash_attention=False, 
                  sl_warp=None,
                  prefill_size=None,
                  autoregressive=False,
                  auto_tune_sl_warp=False,
                  inter_layer_chunk_size=None,
                  model_name="default",
                  sl_cross = None):
        """
        Estimate the latency of a layer running on a PMCA or PMCA_RED tile.
        Supports attention, layernorm, and gelu.
        """

        if layer.type == "atten":
            sl, head_d = layer.input_shape
            sl_full = getattr(layer, "parent_layer", layer).input_shape[0]
            _, hidden_size = layer.output_shape
            atten_type = getattr(layer, "attention_type", "self")  # default to self-attention if not specified

            attn_estimator = AttentionLatencyEstimator(sl, head_d, hidden_size, gemm_fast=gemm_fast, model_name=model_name, prefill_size=prefill_size, autoregressive=autoregressive)
            attn_estimator.optimize_attention_chunk = optimize_attention_chunk
            attn_estimator.sl_full = sl_full  
            attn_estimator.sl_cross = sl_cross

            latency, chunk_ready_times = attn_estimator.total_latency(
                intra_layer_pipelining=intra_layer_pipelining,
                intra_layer_parallelism=intra_layer_parallelism,
                inter_layer_pipelining=inter_layer_pipelining,
                flash_attention=flash_attention, sl_warp=sl_warp,
                auto_tune_sl_warp=auto_tune_sl_warp, 
                inter_layer_chunk_size=inter_layer_chunk_size,
                sl_cross = sl_cross,
                atten_type = atten_type,
                plot=False
            )
            if inter_layer_chunk_size:
                return latency, chunk_ready_times
            else: 
                return latency
        
        elif layer.type in ("layernorm", "gelu", "softmax", "residual"):
            in_r,in_c = layer.input_shape
            if in_r < PMCA_TILE_CONFIG["num_cores"]:
                in_r = PMCA_TILE_CONFIG["num_cores"]
            
            #model_type = "LayerNorm" if layer.type == "layernorm" else "GeLU"
            if layer.type == "layernorm":
                model_type = "LayerNorm" 
            elif layer.type == "gelu":
                model_type = "GeLU"
            elif layer.type == "softmax":
                model_type = "Softmax"
            elif layer.type == "residual":
                model_type = "ElementwiseSum"

            pmca_model = PMCA_pred_models(M=in_r, N=in_c, K=0)
            pred = pmca_model.predict(model_type=model_type)
            clock = PMCA_RED_TILE_CONFIG["clock"] if gemm_fast else PMCA_TILE_CONFIG["clock"]
            return pred[0] * clock
        elif layer.type in ("gemm"):
            if len(layer.input_shape) > 2:
                M,N,K = layer.input_shape
            else:
                M,N = layer.input_shape
                _,K = layer.output_shape
            if M < PMCA_TILE_CONFIG["num_cores"]:
                M = PMCA_TILE_CONFIG["num_cores"]
            if gemm_fast:
                pmca_model = PMCA_pred_models(M=M, N=K, K=N, gemm_fast=True) 
                gemm_pred = pmca_model.predict(model_type="GEMM")
                return gemm_pred[0]*PMCA_RED_TILE_CONFIG["clock"]
            else:
                pmca_model = PMCA_pred_models(M=M, N=N, K=K, gemm_fast=False)
                gemm_pred = pmca_model.predict(model_type="GEMM")
                return gemm_pred[0]*PMCA_TILE_CONFIG["clock"]
        else:
            raise ValueError("Unsupported layer type for cost estimation: {}".format(layer.type))
    
    @staticmethod 
    def DA_cost(layer,
                  da_type=None,
                  intra_layer_parallelism=False,
                  intra_layer_pipelining=False,
                  inter_layer_pipelining=False,
                  inter_layer_chunk_size=None,
                  optimize_attention_chunk=True,
                  flash_attention=False,
                  model_name="default"):
        """
        Estimate the latency of a layer running on a DA tile.
        Supports layernorm, gelu, softmax.
        """
        if da_type == "DA0":
            if layer.type == "atten":
                sl, head_d = layer.input_shape
                sl_full = getattr(layer, "parent_layer", layer).input_shape[0]
                _, hidden_size = layer.output_shape

                attn_estimator = AttentionLatencyEstimatorDAs(sl, head_d, hidden_size, model_name=model_name, max_size_vect_parall=DA0_TILE_CONFIG["max_parallel"], sl_full=sl_full)
                attn_estimator.optimize_attention_chunk = optimize_attention_chunk
                latency, chunk_ready_times = attn_estimator.total_latency(
                    intra_layer_pipelining=intra_layer_pipelining,
                    intra_layer_parallelism=intra_layer_parallelism,
                    inter_layer_pipelining=inter_layer_pipelining,
                    inter_layer_chunk_size=inter_layer_chunk_size,
                    plot=False
                )
                if inter_layer_chunk_size:
                    return latency, chunk_ready_times
                else: 
                    return latency
            elif layer.type in ("layernorm", "gelu", "softmax"):
                in_r,in_c = layer.input_shape
                if in_r < DA0_TILE_CONFIG["num_cores"]:
                    in_r = DA0_TILE_CONFIG["num_cores"]
                
                if layer.type == "layernorm":
                    # 48ns for every 512 data + 2ns for reading and writing a 512 vector in and out of SRAM
                    pred = DA0_TILE_CONFIG["const_delay"] + math.ceil(in_r * in_c / 512) * math.ceil(192/ DA0_TILE_CONFIG['num_cores'] + 2) 
                elif layer.type == "gelu":
                    # 3 pipe stages, 2 ns each + 1ns stage at the beginning and at the end for reading/writing data in SRAM
                    pred = DA0_TILE_CONFIG["const_delay"] + 3*2 + 2 + (math.ceil(in_r*in_c / (128/DA0_TILE_CONFIG['num_cores'])) -1)
                elif layer.type == "softmax":
                    # Reuse AttentionLatencyEstimatorDAs.softmax
                    # softmax only needs matrix shape (sl x sl_full); head_dim not used here.
                    est = AttentionLatencyEstimatorDAs(
                    sl=in_r,
                    head_dim=1,                 # dummy, not used by softmax
                    hidden_size=1,              # dummy, not used by softmax
                    model_name=model_name,
                    max_size_vect_parall=DA0_TILE_CONFIG["max_parallel"],
                    sl_full=in_c
                )
                    pred = DA0_TILE_CONFIG["const_delay"] + est.estimate_softmax_latency(in_r, in_c) / DA0_TILE_CONFIG["clock"]
                
                clock = DA0_TILE_CONFIG["clock"]
                return pred * clock
            
            elif layer.type == "gemm":
                # Try to interpret this GEMM as the attention A*V (softmax x V) and reuse estimate_qkv_latency.
                # We need sl, sl_full, head_dim, and number_of_repetition.
                # Prefer explicit attributes if present; otherwise derive and fall back.
                # Expected for AV: input_shape ~ (sl, sl_full), output_shape ~ (sl, head_dim)
                in_r,in_c,head_dim  = layer.input_shape

                max_heads_in_parallel = DA0_TILE_CONFIG["num_cores"] / math.ceil(head_dim / DA0_TILE_CONFIG["max_parallel"])
                num_of_repetition = math.ceil(1/DA0_TILE_CONFIG["num_tiles"]/max_heads_in_parallel)

                est = AttentionLatencyEstimatorDAs(
                    sl=in_r,
                    head_dim=head_dim,
                    hidden_size=1,
                    model_name=model_name,
                    max_size_vect_parall=DA0_TILE_CONFIG["max_parallel"],
                    sl_full=in_c
                )
                pred = (DA0_TILE_CONFIG["const_delay"]
                        + est.estimate_qkv_latency(in_r, in_c, head_dim, num_of_repetition)/ DA0_TILE_CONFIG["clock"])
                clock = DA0_TILE_CONFIG["clock"]
                return pred * clock
            else:
                raise ValueError("Unsupported layer type for cost estimation: {}".format(layer.type))
        else:
            raise ValueError("Unsupported Digital Accelerator type: {}".format(da_type))

    @staticmethod
    def DDR_latencycost(layer):
        SL = layer.input_shape[0]
        out_shape = layer.output_shape[0]
        bandwidth_B_per_ms = DDR_TILE_CONFIG["bandwidth"] * (2**20) / (10**3)
        tot_byte_to_fetch = SL * out_shape * ACIM_TILE_CONFIG['precision']/8
        fetch_latency_ms = tot_byte_to_fetch/bandwidth_B_per_ms
        return fetch_latency_ms
    
    @staticmethod
    def link_latencycost(layer, SL_scale=None):
        # assume layer.output_shape = (n_elements, bit_width)
        if layer.output_shape is tuple():
            n_elem, bit_w = layer.output_shape
            total_bits = n_elem * bit_w  #TODO
        else:
            if len(layer.output_shape)>1:
                n_elem =layer.output_shape[1]
            else :
                n_elem =layer.output_shape[0]
            if SL_scale is not None:
                total_bits = n_elem * SL_scale
            else:
                total_bits = n_elem

        cycles = math.ceil(total_bits*ACIM_TILE_CONFIG["precision"] / LINK_CONFIG["bandwidth_b_per_cycle"])
        return cycles * LINK_CONFIG["clock"]
    
    @staticmethod
    def SRAM_latency(layer):
        SL = layer.input_shape[0]
        out_shape = layer.output_shape[0]
        bandwidth_B_per_ms = SRAM_TILE_CONFIG["bandwidth"] * (2**20) / (10**3)
        tot_byte_to_fetch = SL * out_shape * ACIM_TILE_CONFIG['precision']/8
        fetch_latency_ms = tot_byte_to_fetch/bandwidth_B_per_ms
        return fetch_latency_ms
    
class PowerModel:
    @staticmethod
    def PMCA_RED_powercost(layer):
        return PMCA_RED_TILE_CONFIG["power"]
    
    @staticmethod
    def PMCA_powercost(layer):
        return PMCA_TILE_CONFIG["power"]
    
    @staticmethod
    def DA0_powercost(layer):
        return DA0_TILE_CONFIG["power"]
    
    @staticmethod
    def DDR_powercost(layer):
        return DDR_TILE_CONFIG["power"]
    
    @staticmethod
    def Link_powercost(layer):
        return LINK_CONFIG["power"]
    
    @staticmethod
    def SRAM_powercost(layer):
        return SRAM_TILE_CONFIG["power"]

class MemoryEstimator:
    @staticmethod
    def estimate_memory_kb(layer, sl_full=None, flash_attention=False, chunk_size=None):
        """
        Estimate memory required in KB for a given layer.
        Supports attention, layernorm, and gelu layers.
        """
        layer_type = layer.type
        sl, head_d = layer.input_shape
        _, hidden_size = layer.output_shape
        precision = PMCA_TILE_CONFIG["fp_precision"]
        bytes_per_element = precision / 8

        if layer_type == "atten":
            # Use sl_full from parent if available, else fallback to current sl
            if sl_full is None:
                sl_full = getattr(layer, "parent_layer", layer).input_shape[0]


            # Attention layers: we assume Q, K, V + intermediate GEMM outputs
            size_q = sl * head_d if chunk_size==None else chunk_size* head_d
            size_k = sl_full * head_d
            size_qk_out = sl * sl_full if chunk_size==None else chunk_size* sl_full
            size_qkv_out = sl * head_d if chunk_size==None else chunk_size* head_d

            if flash_attention and sl!=sl_full:
                total_bytes = size_q + size_k + 2* size_qk_out + 3*sl
            else:
                total_bytes = size_q + size_k + size_qk_out

        elif layer_type == "layernorm" or layer_type == "gelu" or layer_type == "residual":
            total_bytes = sl * head_d if chunk_size==None else chunk_size* head_d
        elif layer_type == "gemm":
            total_bytes = sl * head_d + head_d*hidden_size+sl*hidden_size if chunk_size==None else chunk_size* head_d + head_d*hidden_size+chunk_size*hidden_size
        else:
            return 0  # Conservative fallback

        return (total_bytes * bytes_per_element) / 1024  # Convert to KB
    

class AttentionLatencyEstimator:
    def __init__(self, sl, head_dim, hidden_size, model_name="default", gemm_fast=False, sl_full=None, prefill_size=None, autoregressive=False, sl_cross=None):
        self.sl = sl  # Chunked Q length
        self.sl_full = sl_full if sl_full is not None else sl  # Full K/V length
        self.head_dim = head_dim     # dimension per attention head
        self.hidden_size = hidden_size
        self.num_heads = hidden_size//head_dim
        self.model_name = model_name
        self.plot_once = 0
        self.chunk_latency_cache = {}
        self.gemm_fast = gemm_fast
        self.optimize_attention_chunk = False
        self.prefill_size = prefill_size
        self.autoregressive = autoregressive

        #Accelerator units definition
        if self.gemm_fast == True:
            self.GEMM_UNIT = "GEMM_ACC"
        else:
            self.GEMM_UNIT = "CORES"
        

    def estimate_dma_latency(self, M, N):
        """Estimate DMA transfer latency (ms)."""
        dma_bandwidth = PMCA_RED_TILE_CONFIG["dma_bandwidth"]
        size_B = M * N * (PMCA_RED_TILE_CONFIG["fp_precision"] / 8)
        latency = size_B/dma_bandwidth * PMCA_RED_TILE_CONFIG["clock"]
        return latency 
    
    def estimate_softmax_latency(self, M=None, N=None):
        """ Estimate Softmax latency for MxN matrix (ms)."""
        M = M if M is not None else self.sl
        N = N if N is not None else self.sl_full
        if M < PMCA_RED_TILE_CONFIG["num_cores"]:
            M = PMCA_RED_TILE_CONFIG["num_cores"]
        pmca_model = PMCA_pred_models(M=M, N=N, K=0)  
        softmax_pred = pmca_model.predict(model_type="Softmax")
        return softmax_pred[0] * PMCA_RED_TILE_CONFIG["clock"]
    
    def estimate_partialsoftmax_latency(self, M=None, N=None, K=None):
        if M < PMCA_RED_TILE_CONFIG["num_cores"]:
            M = PMCA_RED_TILE_CONFIG["num_cores"]
        """ Estimate Softmax latency for MxN matrix (ms)."""
        pmca_model = PMCA_pred_models(M=M, N=N, K=K)  
        partialsoftmax_pred = pmca_model.predict(model_type="PartialSoftmax")
        return partialsoftmax_pred[0] * PMCA_RED_TILE_CONFIG["clock"]
    
    def estimate_partialscale_latency(self,M=None, N=None, K=None):
        """ Estimate Softmax latency for MxN matrix (ms)."""
        if M < PMCA_RED_TILE_CONFIG["num_cores"]:
            M = PMCA_RED_TILE_CONFIG["num_cores"]
        pmca_model = PMCA_pred_models(M=M, N=N, K=K)  
        partialscale_pred = pmca_model.predict(model_type="PartialScale")
        return partialscale_pred[0] * PMCA_RED_TILE_CONFIG["clock"]
    
    def estimate_finalscale_latency(self, M=None, N=None, K=None):
        """ Estimate Softmax latency for MxN matrix (ms)."""
        if M < PMCA_RED_TILE_CONFIG["num_cores"]:
            M = PMCA_RED_TILE_CONFIG["num_cores"]
        pmca_model = PMCA_pred_models(M=M, N=N, K=K)  
        finalscale_pred = pmca_model.predict(model_type="FinalScale")
        return finalscale_pred[0] * PMCA_RED_TILE_CONFIG["clock"]
    
    def estimate_gemm_latency(self, M, N, K):
        """Estimate GEMM latency:
        - Redmule (gemm_fast = True) : (MxN) * (NxK);
        - Cores (gemm_fast = False) : (MxK) * (KxN).
        """
        if M < PMCA_RED_TILE_CONFIG["num_cores"]:
            M = PMCA_RED_TILE_CONFIG["num_cores"]
        if self.gemm_fast:
            pmca_model = PMCA_pred_models(M=M, N=K, K=N, gemm_fast=True) 
            gemm_pred = pmca_model.predict(model_type="GEMM")
            return gemm_pred[0]*PMCA_RED_TILE_CONFIG["clock"]
        else:
            pmca_model = PMCA_pred_models(M=M, N=N, K=K, gemm_fast=False)
            gemm_pred = pmca_model.predict(model_type="GEMM")
            return gemm_pred[0]*PMCA_TILE_CONFIG["clock"]

    def build_operation_graph_serial(self, *latencies, flash_attention=False, sl_warp=None, sl_cross=None, atten_type=None):
        """
        Build serial schedule for all heads.
        Latencies: GEMM_QK, GEMM_QKV, Softmax, DMA_Q, DMA_K
        """
        gemm_qk, gemm_qkv, softmax_lat, dma_q_lat, dma_k_lat, dma_v_lat, dma_out_lat = latencies
        ops = []
        prev_dma_out = None

        for h in range(self.num_heads):
            op = self._build_head_ops_serial(h, gemm_qk, gemm_qkv, softmax_lat, dma_q_lat, dma_k_lat, dma_v_lat, dma_out_lat,flash_attention, sl_warp, sl_cross, atten_type)
            if prev_dma_out:
                op[0].add_dependency(prev_dma_out)
            ops.extend(op)
            prev_dma_out = op[-1]

        return ops
    
    def _build_head_ops_serial(self, head_idx, gemm_qk, gemm_qkv, softmax_lat, dma_q_lat, dma_k_lat, dma_v_lat, dma_out_lat, flash_attention=False, sl_warp=None,sl_cross=None, atten_type=None):
        """Build a list of ops for a single head in serial."""
        q = f"Q_{head_idx}"
        
        compute_ops = []

        size = lambda m, n: m * n * PMCA_RED_TILE_CONFIG["fp_precision"] / 8 / 1024  # KB

        if atten_type == "cross":
            KV_size = sl_cross if (self.autoregressive) else self.sl_full
        else:
            KV_size = self.prefill_size if self.autoregressive else self.sl_full

        if not flash_attention or KV_size <= sl_warp:
            k = f"K_{head_idx}"
            v = f"V_{head_idx}"
            out_qk = f"OUT_QK_{head_idx}"
            out_qkv = f"OUT_QKV_{head_idx}"

            dma_qk = AttentionOps(f"DMA_QK_H{head_idx}", "DMA", 
                                  self.estimate_dma_latency(self.sl,self.head_dim) + self.estimate_dma_latency(KV_size,self.head_dim),
                                tcdm_alloc=[{"name": q, "size": size(self.sl, self.head_dim)},
                                            {"name": k, "size": size(KV_size, self.head_dim)}])

            gemm1 = AttentionOps(f"GEMM1_H{head_idx}", "GEMM", self.estimate_gemm_latency(self.sl,KV_size,self.head_dim),
                                tcdm_alloc=[{"name": out_qk, "size": size(self.sl, KV_size)}],
                                tcdm_free=[q, k])

            softmax = AttentionOps(f"Softmax_H{head_idx}", "SOFTMAX", self.estimate_softmax_latency(self.sl,KV_size))

            dma_v = AttentionOps(f"DMA_V_H{head_idx}", "DMA", self.estimate_dma_latency(KV_size,self.head_dim),
                                tcdm_alloc=[{"name": v, "size": size(KV_size, self.head_dim)}])

            gemm2 = AttentionOps(f"GEMM2_H{head_idx}", "GEMM", self.estimate_gemm_latency(self.sl,self.head_dim,KV_size),
                                tcdm_alloc=[{"name": out_qkv, "size": size(self.sl, self.head_dim)}],
                                tcdm_free=[v, out_qk])

            dma_out = AttentionOps(f"DMA_OUT_H{head_idx}", "DMA", self.estimate_dma_latency(self.sl,self.head_dim),
                                tcdm_free=[out_qkv])

            # Dependencies
            gemm1.add_dependency(dma_qk)
            softmax.add_dependency(gemm1)
            dma_v.add_dependency(softmax)
            gemm2.add_dependency(softmax)
            gemm2.add_dependency(dma_v)
            dma_out.add_dependency(gemm2)

            return [dma_qk, gemm1, softmax, dma_v, gemm2, dma_out]
        
        else:
            assert sl_warp is not None, "FlashAttention requires sl_warp to be specified"
            num_warps = math.ceil(KV_size / sl_warp)
            warp_outputs = []
            prev_max = f"PREV_MAX_w0_H{head_idx}"
            prev_acc = f"PREV_ACC_w0_H{head_idx}"
            prev_expshift = f"PREV_EXPSHIFT_w0_H{head_idx}"
            prev_out = f"PREV_OUT_PART_w0_H{head_idx}"

            dma_q = AttentionOps(f"DMA_Q_H{head_idx}", "DMA", dma_q_lat,
                                tcdm_alloc=[{"name": q, "size": size(self.sl, self.head_dim)}])
            final_out_scale = AttentionOps(f"Final_Scale_H{head_idx}", "SOFTMAX", self.estimate_finalscale_latency(self.sl,self.head_dim, sl_warp),
                                           tcdm_free=[q, prev_max, prev_acc, prev_expshift, prev_out])       

            prev_partial_scale = None
            
            for w in range(num_warps):
                # Correct warp size
                warp_size = sl_warp if w < num_warps - 1 else KV_size - sl_warp * (num_warps - 1)

                k = f"K_w{w}_H{head_idx}"
                v = f"V_w{w}_H{head_idx}"
                out_qk = f"OUT_QK_w{w}_H{head_idx}"
                out_part = f"OUT_PARTIAL_w{w}_H{head_idx}"
                
                

                dma_k = AttentionOps(f"DMA_K_w{w}_H{head_idx}", "DMA", self.estimate_dma_latency(warp_size, self.head_dim),
                                    tcdm_alloc=[{"name": k, "size": size(warp_size, self.head_dim)}])

                gemm1 = AttentionOps(f"GEMM1_w{w}_H{head_idx}", "GEMM", self.estimate_gemm_latency(self.sl, warp_size, self.head_dim),
                                    tcdm_alloc=[{"name": out_qk, "size": size(self.sl, warp_size)}],
                                    tcdm_free=[k])
                if w==0:
                    partial_softmax = AttentionOps(f"Softmax_w{w}_H{head_idx}", "SOFTMAX", 
                                                self.estimate_partialsoftmax_latency(self.sl,self.head_dim, warp_size),        
                                                tcdm_alloc=[{"name": prev_max, "size": size(self.sl,1)},
                                                            {"name": prev_acc, "size": size(self.sl,1)},
                                                            {"name": prev_expshift, "size": size(self.sl,1)}])
                else:
                    partial_softmax = AttentionOps(f"Softmax_w{w}_H{head_idx}", "SOFTMAX", self.estimate_partialsoftmax_latency(self.sl,self.head_dim, warp_size))         #TODO

                dma_v = AttentionOps(f"DMA_V_w{w}_H{head_idx}", "DMA", self.estimate_dma_latency(warp_size,self.head_dim),
                                    tcdm_alloc=[{"name": v, "size": size(warp_size, self.head_dim)}])
                gemm2 = AttentionOps(f"GEMM2_w{w}_H{head_idx}", "GEMM", self.estimate_gemm_latency(self.sl, self.head_dim, warp_size),
                                    tcdm_alloc=[{"name": out_part, "size": size(self.sl, self.head_dim)}],
                                    tcdm_free=[v, out_qk])
                if w==0:
                    partial_scale = AttentionOps(f"Partial_Scale_w{w}_H{head_idx}", "SOFTMAX", 
                                            self.estimate_partialscale_latency(self.sl,self.head_dim, warp_size),                 
                                            tcdm_alloc=[{"name": prev_out, "size": size(self.sl, self.head_dim)}],
                                            tcdm_free=[out_part])
                else:
                    partial_scale = AttentionOps(f"Partial_Scale_w{w}_H{head_idx}", "SOFTMAX",
                                            self.estimate_partialscale_latency(self.sl,self.head_dim, warp_size),                  
                                            tcdm_free=[out_part])
                    
                # Connect ops within warp
                if prev_partial_scale != None:
                    dma_k.add_dependency(prev_partial_scale)
                
                gemm1.add_dependency(dma_q)
                gemm1.add_dependency(dma_k)
                partial_softmax.add_dependency(gemm1)
                dma_v.add_dependency(partial_softmax)
                gemm2.add_dependency(partial_softmax)
                gemm2.add_dependency(dma_v)
                partial_scale.add_dependency(gemm2)

                # Final scale depends on all warp outputs
                final_out_scale.add_dependency(partial_scale)

                warp_outputs.append(partial_scale)
                compute_ops.extend([dma_k, gemm1, partial_softmax, dma_v, gemm2, partial_scale])

                prev_partial_scale = partial_scale
            
            dma_out = AttentionOps(f"DMA_OUT_H{head_idx}", "DMA", dma_out_lat,
                                tcdm_free=[f"OUT_PARTIAL_w{w}_H{head_idx}" for w in range(num_warps)])

            dma_out.add_dependency(final_out_scale)
            return [dma_q] + compute_ops + [final_out_scale, dma_out]

    def build_attention_ops_for_chunk(self, chunk_size, flash_attention=False, sl_warp=None, sl_cross=None, atten_type=None):
        ops, chunk_ops = [], []
        num_chunks = math.ceil(self.sl / chunk_size)
        KB = PMCA_RED_TILE_CONFIG["fp_precision"] / 8 / 1024  # Convert to KB
        prev_dma_qk_or_q = None
        prev_dma_v = None
        if atten_type == "cross":
            KV_size = sl_cross if (self.autoregressive) else self.sl_full
        else:
            KV_size = self.prefill_size if self.autoregressive else self.sl_full

        for chunk_idx in range(num_chunks):
            chunk_ops_this_chunk = []
            sl_chunk = chunk_size if chunk_idx < num_chunks - 1 else self.sl - chunk_size * (num_chunks - 1)

            for head_idx in range(self.num_heads):
                q_name = f"Q_chunk{chunk_idx}_H{head_idx}"
                size_q = sl_chunk * self.head_dim * KB

                if not flash_attention or KV_size <= sl_warp:
                    # --- Standard Attention: Q+K together ---
                    k_name = f"K_H{head_idx}"
                    v_name = f"V_H{head_idx}"
                    out_qk = f"OUT_QK_chunk{chunk_idx}_H{head_idx}"
                    out_qkv = f"OUT_QKV_chunk{chunk_idx}_H{head_idx}"

                    dma_qk = AttentionOps(
                        f"DMA_QK_chunk{chunk_idx}_H{head_idx}", "DMA",
                        duration=self.estimate_dma_latency(sl_chunk, self.head_dim) + self.estimate_dma_latency(KV_size, self.head_dim),
                        tcdm_alloc=[{"name": q_name, "size": size_q},
                                    {"name": k_name, "size": KV_size * self.head_dim * KB}],
                        tcdm_free=[]
                    )

                    # Head-wise dependency
                    if head_idx == 0:
                        pass
                    elif head_idx % 2 == 1:
                        dma_qk.add_dependency(prev_dma_qk_or_q)
                    else :
                        dma_qk.add_dependency(prev_dma_v)

                    gemm1 = AttentionOps(
                        f"GEMM1_chunk{chunk_idx}_H{head_idx}", "GEMM",
                        duration=self.estimate_gemm_latency(sl_chunk, KV_size, self.head_dim),
                        tcdm_alloc=[{"name": out_qk, "size": sl_chunk * KV_size * KB}],
                        tcdm_free=[q_name, k_name]
                    )

                    softmax = AttentionOps(
                        f"Softmax_chunk{chunk_idx}_H{head_idx}", "SOFTMAX",
                        duration=self.estimate_softmax_latency(sl_chunk, KV_size)
                    )

                    dma_v = AttentionOps(
                        f"DMA_V_chunk{chunk_idx}_H{head_idx}", "DMA",
                        duration=self.estimate_dma_latency(KV_size, self.head_dim),
                        tcdm_alloc=[{"name": v_name, "size": KV_size * self.head_dim * KB}],
                        tcdm_free=[]
                    )

                    gemm2 = AttentionOps(
                        f"GEMM2_chunk{chunk_idx}_H{head_idx}", "GEMM",
                        duration=self.estimate_gemm_latency(sl_chunk, self.head_dim, KV_size),
                        tcdm_alloc=[{"name": out_qkv, "size": sl_chunk * self.head_dim * KB}],
                        tcdm_free=[v_name, out_qk]
                    )

                    dma_out = AttentionOps(
                        f"DMA_OUT_chunk{chunk_idx}_H{head_idx}", "DMA",
                        duration=self.estimate_dma_latency(sl_chunk, self.head_dim),
                        tcdm_free=[out_qkv]
                    )

                    gemm1.add_dependency(dma_qk)
                    softmax.add_dependency(gemm1)
                    dma_v.add_dependency(gemm1)
                    gemm2.add_dependency(softmax)
                    gemm2.add_dependency(dma_v)
                    dma_out.add_dependency(gemm2)

                    chunk_ops_this_chunk.extend([dma_qk, gemm1, softmax, dma_v, gemm2, dma_out])
                    prev_dma_qk_or_q = dma_qk
                    prev_dma_v = dma_v

                else:
                    # --- FlashAttention: Q and K separate, warp-style ---
                    assert sl_warp is not None, "FlashAttention requires sl_warp"
                    num_warps = math.ceil(KV_size / sl_warp)
                    prev_partial = None

                     # Buffers for accumulation across warps
                    prev_max = f"PREV_MAX_chunk{chunk_idx}_H{head_idx}"
                    prev_acc = f"PREV_ACC_chunk{chunk_idx}_H{head_idx}"
                    prev_expshift = f"PREV_EXPSHIFT_chunk{chunk_idx}_H{head_idx}"
                    prev_out = f"PREV_OUT_PART_chunk{chunk_idx}_H{head_idx}"

                    final_out = AttentionOps(
                        f"Final_Scale_chunk{chunk_idx}_H{head_idx}", "SOFTMAX",
                        duration=self.estimate_finalscale_latency(sl_chunk,self.head_dim, sl_warp),
                        tcdm_free=[prev_max, prev_acc, prev_expshift, prev_out]
                    )

                    dma_q = AttentionOps(
                        f"DMA_Q_chunk{chunk_idx}_H{head_idx}", "DMA",
                        duration=self.estimate_dma_latency(sl_chunk, self.head_dim),
                        tcdm_alloc=[{"name": q_name, "size": size_q}],
                        tcdm_free=[]
                    )

                    # Head-wise dependency for DMA_Q
                    if head_idx == 0:
                        pass
                    elif head_idx % 2 == 1:
                        dma_q.add_dependency(prev_dma_qk_or_q)
                    else:
                        dma_q.add_dependency(prev_dma_v)

                    warp_ops = []

                    for warp_idx in range(num_warps):
                        warp_size = sl_warp if warp_idx < num_warps - 1 else KV_size - sl_warp * (num_warps - 1)

                        k_name = f"K_w{warp_idx}_chunk{chunk_idx}_H{head_idx}"
                        v_name = f"V_w{warp_idx}_chunk{chunk_idx}_H{head_idx}"
                        out_qk = f"OUT_QK_w{warp_idx}_chunk{chunk_idx}_H{head_idx}"
                        out_part = f"OUT_PARTIAL_w{warp_idx}_chunk{chunk_idx}_H{head_idx}"

                        dma_k = AttentionOps(
                            f"DMA_K_w{warp_idx}_chunk{chunk_idx}_H{head_idx}", "DMA",
                            duration=self.estimate_dma_latency(warp_size, self.head_dim),
                            tcdm_alloc=[{"name": k_name, "size": warp_size * self.head_dim * KB}]
                        )

                        if warp_idx==(num_warps-1):
                            gemm1 = AttentionOps(
                                f"GEMM1_w{warp_idx}_chunk{chunk_idx}_H{head_idx}", "GEMM",
                                duration=self.estimate_gemm_latency(sl_chunk, warp_size, self.head_dim),
                                tcdm_alloc=[{"name": out_qk, "size": sl_chunk * warp_size * KB}],
                                tcdm_free=[k_name, q_name]
                            )
                        else:
                            gemm1 = AttentionOps(
                                f"GEMM1_w{warp_idx}_chunk{chunk_idx}_H{head_idx}", "GEMM",
                                duration=self.estimate_gemm_latency(sl_chunk, warp_size, self.head_dim),
                                tcdm_alloc=[{"name": out_qk, "size": sl_chunk * warp_size * KB}],
                                tcdm_free=[k_name]
                            )

                        # Softmax with extra buffers for warp 0
                        if warp_idx == 0:
                            partial_softmax = AttentionOps(
                                f"Softmax_w{warp_idx}_chunk{chunk_idx}_H{head_idx}", "SOFTMAX",
                                duration=self.estimate_partialsoftmax_latency(sl_chunk,self.head_dim, warp_size), 
                                tcdm_alloc=[
                                    {"name": prev_max, "size": sl_chunk * 1 * KB},
                                    {"name": prev_acc, "size": sl_chunk * 1 * KB},
                                    {"name": prev_expshift, "size": sl_chunk * 1 * KB}
                                ]
                            )
                        else:
                            partial_softmax = AttentionOps(
                                f"Softmax_w{warp_idx}_chunk{chunk_idx}_H{head_idx}", "SOFTMAX",
                                duration=self.estimate_partialsoftmax_latency(sl_chunk,self.head_dim, warp_size), 
                            )

                        dma_v = AttentionOps(
                            f"DMA_V_w{warp_idx}_chunk{chunk_idx}_H{head_idx}", "DMA",
                            duration=self.estimate_dma_latency(warp_size, self.head_dim),
                            tcdm_alloc=[{"name": v_name, "size": warp_size * self.head_dim * KB}]
                        )

                        gemm2 = AttentionOps(
                            f"GEMM2_w{warp_idx}_chunk{chunk_idx}_H{head_idx}", "GEMM",
                            duration=self.estimate_gemm_latency(sl_chunk, self.head_dim, warp_size),
                            tcdm_alloc=[{"name": out_part, "size": sl_chunk * self.head_dim * KB}],
                            tcdm_free=[v_name, out_qk]
                        )

                        if warp_idx == 0:
                            partial_scale = AttentionOps(
                                f"Partial_Scale_w{warp_idx}_chunk{chunk_idx}_H{head_idx}", "SOFTMAX",
                                duration=self.estimate_partialscale_latency(sl_chunk,self.head_dim, warp_size),
                                tcdm_alloc=[{"name": prev_out, "size": sl_chunk * self.head_dim * KB}],
                                tcdm_free=[out_part]
                            )
                        else:
                            partial_scale = AttentionOps(
                                f"Partial_Scale_w{warp_idx}_chunk{chunk_idx}_H{head_idx}", "SOFTMAX",
                                duration=self.estimate_partialscale_latency(sl_chunk,self.head_dim, warp_size),
                                tcdm_free=[out_part]
                            )

                        # Warp-wise dependency
                        if warp_idx == 0:
                            dma_k.add_dependency(dma_q)
                        elif warp_idx % 2 == 1 and prev_dma_k:
                            dma_k.add_dependency(prev_dma_k)
                        elif prev_dma_v:
                            dma_k.add_dependency(prev_dma_v)

                        gemm1.add_dependency(dma_q)
                        gemm1.add_dependency(dma_k)
                        partial_softmax.add_dependency(gemm1)
                        dma_v.add_dependency(gemm1)
                        gemm2.add_dependency(partial_softmax)
                        gemm2.add_dependency(dma_v)
                        partial_scale.add_dependency(gemm2)
                        final_out.add_dependency(partial_scale)

                        warp_ops.extend([dma_k, gemm1, partial_softmax, dma_v, gemm2, partial_scale])
                        prev_dma_k = dma_k
                        prev_dma_v = dma_v
                        prev_partial = partial_scale

                    dma_out = AttentionOps(
                        f"DMA_OUT_chunk{chunk_idx}_H{head_idx}", "DMA",
                        duration=self.estimate_dma_latency(sl_chunk, self.head_dim),
                        tcdm_free=[f"OUT_PARTIAL_w{w}_chunk{chunk_idx}_H{head_idx}" for w in range(num_warps)]
                    )
                    dma_out.add_dependency(final_out)

                    chunk_ops_this_chunk.extend([dma_q] + warp_ops + [final_out, dma_out])
                    prev_dma_qk_or_q = dma_v
                    prev_dma_v = dma_v

            # --- Set dependencies across chunks ---
            if chunk_ops:
                num_heads = self.num_heads
                
                if num_heads == 1:
                    if flash_attention:
                        prev_dma_qk_ops = [op for op in reversed(chunk_ops) if op.name.startswith("DMA_K")]
                    else:
                        prev_dma_qk_ops = [op for op in reversed(chunk_ops) if op.name.startswith("DMA_QK")]

                    prev_dma_v_ops  = [op for op in reversed(chunk_ops) if op.name.startswith("DMA_V")]

                    if chunk_idx % 2 == 0:
                        # Even chunk index: depend on last DMA_V
                        last_dma_v = prev_dma_v_ops[0]
                        for op in chunk_ops_this_chunk:
                            if op.name.startswith("DMA_QK") or op.name.startswith("DMA_Q") :
                                op.add_dependency(last_dma_v)
                    else:
                        # Odd chunk index: depend on last DMA_V
                        last_dma_qk = prev_dma_qk_ops[0]
                        for op in chunk_ops_this_chunk:
                            if op.name.startswith("DMA_QK") or op.name.startswith("DMA_Q") :
                                op.add_dependency(last_dma_qk)
                else:
                    # Multiple heads => must wait for second-to-last DMA_OUT
                    dma_out_ops = [op for op in chunk_ops if op.name.startswith("DMA_OUT")]
                    assert len(dma_out_ops) >= 2, "Not enough DMA_OUT ops to find second-to-last!"
                    second_last_dma_out = dma_out_ops[-2]
                    for op in chunk_ops_this_chunk:
                        if op.name.startswith("DMA_QK")  or op.name.startswith("DMA_Q") :
                            op.add_dependency(second_last_dma_out)

            ops.extend(chunk_ops_this_chunk)
            chunk_ops.extend(chunk_ops_this_chunk)

        return ops


    def schedule_sequential_attention(self, ops):
        current_time = 0.0
        scheduled_ops = []
        tcdm_capacity = PMCA_RED_TILE_CONFIG["tcdm_capacity"]
        tcdm_usage = []
        memory_trace = []

        log_filename = f"outputs/Attention/{self.model_name}_attention_tcdm_sequential_schedule_debug.txt"
        log_file = open(log_filename, "w")

        def free_expired_memory(now):
            expired = [mem for mem in tcdm_usage if (mem["free_time"] is not None) and (mem["free_time"] <= now)]
            if expired:
                print(f"[{now:.3f} ms] \u27a1\ufe0f  Freeing memory:", file=log_file)
                for mem in expired:
                    print(f"    - {mem['name']:10s} ({mem['size']:.2f} KB)", file=log_file)
            tcdm_usage[:] = [mem for mem in tcdm_usage if (mem["free_time"] is None) or (mem["free_time"] > now)]

        def available_tcdm():
            return tcdm_capacity - sum(mem["size"] for mem in tcdm_usage)

        for op in ops:
            while True:
                free_expired_memory(current_time)

                required_mem = sum(alloc["size"] for alloc in op.tcdm_alloc)
                if required_mem <= available_tcdm() and op.is_ready(current_time):
                    break  # We can schedule now

                # Check for next unlock event
                future_times = [d.end_time for d in op.dependencies if d.end_time > current_time]
                future_frees = [mem["free_time"] for mem in tcdm_usage if mem["free_time"] is not None and mem["free_time"] > current_time]
                next_event = min(future_times + future_frees, default=None)

                if next_event is None:
                    print("Deadlock: repeated scheduling order", file=log_file)
                    print(f"[{current_time:.3f} ms] Waiting ops: {op}", file=log_file)
                    log_file.close()
                    return scheduled_ops, float("inf"), memory_trace
                    #raise RuntimeError(f"Deadlock: cannot schedule {op.name} and no future events exist.")
                current_time = next_event

            # --- Schedule operation ---
            op.start_time = current_time
            op.end_time = current_time + op.duration
            scheduled_ops.append(op)

            print(f"[{current_time:.3f} ms] Scheduled {op.name:15s} on {op.op_type:8s} @ {op.start_time:.3f} dur={op.duration:.3f}", file=log_file)

            # --- Allocate memory
            for alloc in op.tcdm_alloc:
                tcdm_usage.append({
                    "name": alloc["name"],
                    "size": alloc["size"],
                    "free_time": None
                })

            for entry in tcdm_usage:
                if entry["name"] in op.tcdm_free:
                    entry["free_time"] = op.end_time

            current_time = op.end_time
            memory_trace.append((current_time, sum(mem["size"] for mem in tcdm_usage)))

        free_expired_memory(current_time)
        log_file.close()
        return scheduled_ops, max(op.end_time for op in scheduled_ops), memory_trace

    def schedule_attention_serial_with_chunks(self, flash_attention=False, sl_warp=None, inter_layer_pipelining=False, inter_layer_chunk_size=None, sl_cross=None, atten_type=None):
        """
        Serial version of attention scheduling with chunks.
        Used when inter-layer pipelining is enabled but not intra-layer pipelining.
        """
        if inter_layer_pipelining:
            chunk_size = inter_layer_chunk_size
        else: 
            chunk_size = self.sl
        
        ops = self.build_attention_ops_for_chunk(chunk_size, flash_attention, sl_warp, sl_cross, atten_type)
        scheduled_ops, total_latency, memory_trace = self.schedule_sequential_attention(ops)
        chunk_ready_times = self._extract_chunk_ready_times(scheduled_ops)
        return scheduled_ops, total_latency, chunk_size, memory_trace, chunk_ready_times

    def schedule_attention_pipeline(self, ops, skip_inf_lat=False):
        current_time = 0.0
        scheduled_ops = []
        state = {
            "gemm_busy_until": -1,
            "dma_busy_until": -1,
            "cores_busy_until": -1,
        }
        tcdm_capacity = PMCA_RED_TILE_CONFIG["tcdm_capacity"]
        tcdm_usage = []
        memory_trace = []
        running_ops = []
        rollback_window = 3

        log_file = open(f"outputs/Attention/{self.model_name}_attention_tcdm_pipeline_schedule_debug.txt", "w")

        max_idle_time = 5.0
        idle_time = 0.0

        # -------------------------------------------------------------------
        # keep track of every "snapshot" of scheduled_ops (by name order)
        seen_schedules = []  # list of tuples, each tuple is (op.name, op.name, ...)
        seen_set = set()     # a set of those tuples for quick membership check
        # -------------------------------------------------------------------

        def free_expired_memory(now):
            expired = [mem for mem in tcdm_usage if mem["free_time"] is not None and mem["free_time"] <= now]
            for mem in expired:
                print(f"[{now:.3f} ms] \u27a1\ufe0f  Freeing {mem['name']:10s} ({mem['size']:.2f} KB)", file=log_file)
            tcdm_usage[:] = [mem for mem in tcdm_usage if mem["free_time"] is None or mem["free_time"] > now]

        def available_tcdm():
            return tcdm_capacity - sum(mem["size"] for mem in tcdm_usage)

        def score_op_priority(op):
            alloc = sum(a["size"] for a in op.tcdm_alloc)
            frees = sum(mem["size"] for mem in tcdm_usage if mem["name"] in op.tcdm_free)
            if alloc > available_tcdm():
                return float('-inf')

            # Extract chunk and head index from op name
            chunk_id, head_id, warp_id = 999, 999, 999
            if "chunk" in op.name and "_H" in op.name:
                try:
                    chunk_part = re.search(r"chunk(\d+)", op.name)
                    head_part = re.search(r"_H(\d+)", op.name)
                    warp_part = re.search(r"_w(\d+)", op.name)

                    if chunk_part:
                        chunk_id = int(chunk_part.group(1))
                    if head_part:
                        head_id = int(head_part.group(1))
                    if warp_part:
                        warp_id = int(warp_part.group(1))
                except Exception:
                    pass

            # Prioritize freeing memory
            score = frees - alloc

            # Strong priority: lower chunk, then head, then warp
            score += (10000 - 1000 * chunk_id - 100 * head_id - warp_id)

            # Slight bonus to GEMM2 to free buffers sooner
            if "GEMM2" in op.name:
                score += 10

            return score


        def get_ready_ops(current_time):
            return [op for op in ops if op.start_time is None and op.is_ready(current_time)]

        def rollback_and_reschedule():
            for i in range(1, rollback_window + 1):
                if len(scheduled_ops) < i:
                    break

                rollback_op = scheduled_ops[-1] # scheduled_ops[-i]
                print(f"[{current_time:.3f} ms] \U0001f504 Rolling back {rollback_op.name}", file=log_file)
                scheduled_ops.remove(rollback_op)
                rollback_op.start_time = None
                rollback_op.end_time = None
                tcdm_usage[:] = [mem for mem in tcdm_usage if mem["name"] not in [a["name"] for a in rollback_op.tcdm_alloc]]

                ready_now = get_ready_ops(current_time)
                ready_now = [op for op in ready_now if op != rollback_op]
                ready_now.sort(key=score_op_priority, reverse=True)

                for candidate in ready_now:
                    if self.GEMM_UNIT == "GEMM_ACC":
                        if candidate.op_type == "GEMM" and current_time < state["gemm_busy_until"]:
                            continue
                    elif self.GEMM_UNIT == "CORES":
                        if candidate.op_type == "GEMM" and current_time < state["cores_busy_until"]:
                            continue
                    if candidate.op_type == "DMA" and current_time < state["dma_busy_until"]:
                        continue
                    if candidate.op_type == "SOFTMAX" and current_time < state["cores_busy_until"]:
                        continue
                    if sum(a["size"] for a in candidate.tcdm_alloc) > available_tcdm():
                        continue

                    candidate.start_time = current_time
                    candidate.end_time = current_time + candidate.duration
                    scheduled_ops.append(candidate)
                    if candidate.op_type == "GEMM":
                        if self.GEMM_UNIT == "GEMM_ACC":
                            state["gemm_busy_until"] = candidate.end_time
                        else :
                            state["cores_busy_until"] = candidate.end_time
                    if candidate.op_type == "DMA":
                        state["dma_busy_until"] = candidate.end_time
                    if candidate.op_type == "SOFTMAX":
                        state["cores_busy_until"] = candidate.end_time
                    for alloc in candidate.tcdm_alloc:
                        print(f"[{current_time:.3f} ms] \u27a1\ufe0f  Realloc {alloc['name']:10s} ({alloc['size']:.2f} KB)", file=log_file)
                        tcdm_usage.append({
                            "name": alloc["name"],
                            "size": alloc["size"],
                            "free_time": None
                        })
                    for entry in tcdm_usage:
                        if entry["name"] in candidate.tcdm_free:
                            entry["free_time"] = candidate.end_time
                    print(f"[{current_time:.3f} ms] \u2705 Rescheduled {candidate.name}", file=log_file)
                    return True

            return False

        while True:
            free_expired_memory(current_time)
            ready_ops = get_ready_ops(current_time)
            ready_ops.sort(key=score_op_priority, reverse=True)

            something_scheduled = False
            for op in ready_ops:
                if self.GEMM_UNIT == "GEMM_ACC":
                    if op.op_type == "GEMM" and current_time < state["gemm_busy_until"]:
                        continue
                elif self.GEMM_UNIT == "CORES":
                    if op.op_type == "GEMM" and current_time < state["cores_busy_until"]:
                        continue
                if op.op_type == "DMA" and current_time < state["dma_busy_until"]:
                    continue
                if op.op_type == "SOFTMAX" and current_time < state["cores_busy_until"]:
                    continue
                if sum(alloc["size"] for alloc in op.tcdm_alloc) > available_tcdm():
                    continue

                op.start_time = current_time
                op.end_time = current_time + op.duration
                scheduled_ops.append(op)

                if op.op_type == "GEMM":
                    if self.GEMM_UNIT == "GEMM_ACC":
                        state["gemm_busy_until"] = op.end_time
                    else:
                        state["cores_busy_until"] = op.end_time
                if op.op_type == "DMA":
                    state["dma_busy_until"] = op.end_time
                if op.op_type == "SOFTMAX":
                    state["cores_busy_until"] = op.end_time

                print(f"[{current_time:.3f} ms] Scheduled {op.name:15s} on {op.op_type:8s} @ {op.start_time:.3f} dur={op.duration:.3f}", file=log_file)
                for alloc in op.tcdm_alloc:
                    print(f"[{current_time:.3f} ms] \u27a1\ufe0f  Allocating {alloc['name']:10s} ({alloc['size']:.2f} KB)", file=log_file)
                    tcdm_usage.append({
                        "name": alloc["name"],
                        "size": alloc["size"],
                        "free_time": None
                    })
                for mem in tcdm_usage:
                    if mem["name"] in op.tcdm_free:
                        mem["free_time"] = op.end_time

                something_scheduled = True
                break

            memory_trace.append((current_time, sum(m["size"] for m in tcdm_usage)))

            if not get_ready_ops(current_time) and all(op.end_time <= current_time for op in scheduled_ops):
                break

            if not something_scheduled:
                next_events = [
                    t for t in [state["gemm_busy_until"], state["dma_busy_until"], state["cores_busy_until"]] + 
                    [op.end_time for op in scheduled_ops if op.end_time > current_time] + 
                    [mem["free_time"] for mem in tcdm_usage if mem["free_time"] and mem["free_time"] > current_time]
                ]
                next_events = [t for t in next_events if t > current_time]

                if next_events:
                    current_time = min(next_events)
                    idle_time = 0.0
                else:
                    current_sequence = tuple(op.name for op in scheduled_ops)
                    if current_sequence in seen_set:
                        if not skip_inf_lat:
                            print("Deadlock: repeated scheduling order", file=log_file)
                            waiting_ops = [op.name for op in get_ready_ops(current_time)]
                            print(f"[{current_time:.3f} ms] Waiting ops: {waiting_ops}", file=log_file)
                            log_file.close()
                            return scheduled_ops, float("inf"), memory_trace
                        else:
                            raise RuntimeError("Deadlock: repeated scheduling order")
                    # 3) otherwise, record it and continue with rollback
                    seen_set.add(current_sequence)
                    seen_schedules.append(current_sequence)

                    print(">>> Attempting rollback due to deadlock", file=log_file)
                    if not rollback_and_reschedule():
                        if not skip_inf_lat:
                            print("Deadlock: rollback failed to resolve.", file=log_file)
                            waiting_ops = [op.name for op in get_ready_ops(current_time)]
                            print(f"[{current_time:.3f} ms] Waiting ops: {waiting_ops}", file=log_file)
                            log_file.close()
                            return scheduled_ops, float("inf"), memory_trace
                        else:
                            raise RuntimeError("Deadlock: rollback failed to resolve.")
                        
            else:
                idle_time = 0.0
                

            if idle_time > max_idle_time:
                raise RuntimeError(f"Deadlock detected after {idle_time:.3f} ms idle!")

        log_file.close()
        return scheduled_ops, max(op.end_time for op in scheduled_ops), memory_trace

    

    

    def schedule_attention_pipeline_with_chunks(self, flash_attention=False, sl_warp=None, inter_layer_chunk_size=None, sl_cross=None, atten_type=None):
        """
        Try multiple chunk sizes and return the one with lowest latency.
        """
            # --- Fast path: use fixed chunk size from inter-layer pipelining --- can be merged with next if , if inter_layer_chunk_size is sent=SL when inter_layer pipeline is False
        if not self.optimize_attention_chunk and inter_layer_chunk_size is not None:
            ops = self.build_attention_ops_for_chunk(inter_layer_chunk_size, flash_attention, sl_warp, sl_cross, atten_type)
            scheduled_ops, total_latency, memory_trace = self.schedule_attention_pipeline(ops)
            chunk_ready_times = self._extract_chunk_ready_times(scheduled_ops)
            return scheduled_ops, total_latency, inter_layer_chunk_size, memory_trace, chunk_ready_times

        if not self.optimize_attention_chunk:
            # Quick path: No optimization, use SL as one chunk
            ops = self.build_attention_ops_for_chunk(self.sl, flash_attention, sl_warp,  sl_cross, atten_type)
            scheduled_ops, total_latency, memory_trace = self.schedule_attention_pipeline(ops)
            chunk_ready_times = self._extract_chunk_ready_times(scheduled_ops)

            if flash_attention:
                log_msg = f" sl_warp {sl_warp}: total latency = {total_latency:.3f} ms"
            else:
                log_msg = f"total latency = {total_latency:.3f} ms"
            print(log_msg)
            return scheduled_ops, total_latency, self.sl, memory_trace, chunk_ready_times

        chunk_candidates = self.generate_chunk_candidates(self.sl)

        best = (None, float('inf'), None, None)

        #cache_key = (self.sl, self.head_dim, self.hidden_size, self.gemm_fast)
        cache_key = (self.sl, self.head_dim, self.hidden_size, self.gemm_fast, flash_attention, sl_warp)
        global global_chunk_latency_cache

        log_filename = f"outputs/Attention/{self.model_name}_attention_tcdm_schedule_chunked_debug{'_FA' if flash_attention==True else ''}{'_inter_layer_chunk' if inter_layer_chunk_size != None else ''}{inter_layer_chunk_size if inter_layer_chunk_size != None else ''}.txt"
        log_file = open(log_filename, "a")

        print(f"[Chunk Optimization] Model: {self.model_name}, SL: {self.sl}, FlashAttention: {flash_attention}, SL_WARP: {sl_warp}", file=log_file)

        if  cache_key in global_chunk_latency_cache:
            print(f"[Cache] Using cached result for sl_warp={sl_warp}", file=log_file)
            return global_chunk_latency_cache[cache_key]

        for chunk_size in chunk_candidates:
            # Build ops for current chunk size
            ops = self.build_attention_ops_for_chunk(chunk_size,flash_attention, sl_warp, sl_cross, atten_type)

            # Run normal scheduling (your existing code)
            sched, lat, mem = self.schedule_attention_pipeline(ops, skip_inf_lat=True)

            if flash_attention:
                log_msg = f"Chunk size {chunk_size}, sl_warp {sl_warp}: total latency = {lat:.3f} ms"
            else:
                log_msg = f"Chunk size {chunk_size}: total latency = {lat:.3f} ms"

            print(log_msg)
            print(log_msg, file=log_file)
            ###-------------------
            if lat == float("inf"):
                continue
            ###-------------------

            chunk_ready_times = self._extract_chunk_ready_times(sched)

            if lat < best[1]:
                best = (sched, lat, chunk_size, mem, chunk_ready_times)
            
        log_msg = f"[Best Chunk] Size = {best[2]}, latency = {best[1]:.3f} ms"
        print(log_msg)
        print(log_msg, file=log_file)
        

        global_chunk_latency_cache[cache_key] = best
        log_file.close()
        return best
    
    def _extract_chunk_ready_times(self, scheduled_ops):
        """
        Parse all scheduled ops and return the latest end time among all DMA_OUT_chunk* ops
        per chunk (across all heads).
        Returns: {chunk_idx: max_end_time_across_heads}.
        """
        import re
        from collections import defaultdict

        chunk_to_max_time = defaultdict(float)
        chunk_to_head_times = defaultdict(list)

        for op in scheduled_ops:
            if op.name.startswith("DMA_OUT_chunk"):
                match = re.search(r"chunk(\d+)_H(\d+)", op.name)
                if match:
                    chunk_id = int(match.group(1))
                    head_id = int(match.group(2))
                    chunk_to_head_times[chunk_id].append(op.end_time)

        for chunk_id, times in chunk_to_head_times.items():
            chunk_to_max_time[chunk_id] = max(times)

        return chunk_to_max_time


    def generate_chunk_candidates(self, sl):
        """
        Generate chunk sizes that are multiples of 16 and <= SL.
        """
        return [8] + [x for x in range(16, sl+1, 16)]

    def tune_best_sl_warp(self, gemm_lat_QK, gemm_lat_QKV, softmax_lat, 
                      dma_lat_Q, dma_lat_K, dma_lat_V, dma_out_lat, pipelining=False, parallelism=False,
                      inter_layer_pipelining=False, inter_layer_chunk_size=None, atten_type=None, sl_cross=None,
                      plot_curve=False, verbose=False):
        """
        Automatically find the best sl_warp value to minimize attention latency.
        Uses a coarse-to-fine binary-like search, with:
        - Coarse search: large step (power-of-2)
        - Fine search: local refinement (smaller step)
        - TCDM-aware filtering of invalid warp sizes
        - Fair non-flash attention baseline when warp == sl_full
        - Early termination if fine search shows no improvement
        Returns: best_warp, best_latency, best_schedule, best_memory_trace
        """
        latencies = {}
        visited = set()
        tcdm_skipped = []
        chunk_ready_times = {}
        
        dummy_layer = type("Dummy", (), {})()
        dummy_layer.type = "atten"
        dummy_layer.input_shape = (self.sl, self.head_dim)
        dummy_layer.output_shape = (self.sl, self.hidden_size)
        dummy_layer.parent_layer = type("Parent", (), {"input_shape": (self.sl_full, self.head_dim)})
        
        # Cache to avoid recomputing latency
        global global_warp_latency_cache
        
        sl_to_use = self.sl_full if atten_type != "cross" else sl_cross

        def measure(warp, sl_to_use=None):
            if warp in visited or warp < 8 or warp > sl_to_use:
                return float('inf'), None, None
            visited.add(warp)

            chunk_size = None

            cache_key = (self.model_name, self.sl, self.head_dim, self.hidden_size, self.gemm_fast, warp, pipelining, inter_layer_chunk_size)
            if cache_key in global_warp_latency_cache:
                if verbose:
                    print(f"[Cache] Using cached latency for warp={warp}")
                lat, sched, mem, chunk_ready_times = global_warp_latency_cache[cache_key]
                latencies[warp] = lat
                return lat, sched, mem, chunk_ready_times

            # --- Check TCDM memory constraint ---
            mem_kb = MemoryEstimator.estimate_memory_kb(dummy_layer, sl_full=warp, flash_attention=True, chunk_size=inter_layer_chunk_size if inter_layer_chunk_size!=None else self.sl)
            if mem_kb > PMCA_RED_TILE_CONFIG["tcdm_capacity"]:
                tcdm_skipped.append(warp)
                if verbose:
                    print(f"  warp={warp:>3}  --> skipped (memory {mem_kb:.1f} KB > TCDM)")
                return float('inf'), None, None, None

            # --- Fair baseline: no flash attention when warp == sl_full ---
            if warp == sl_to_use:
                print(f"Running Standard Attention with warp size equal to SL ({warp})")
                if pipelining and inter_layer_pipelining:
                    sched, lat, chunk_size, mem, chunk_ready_times = self.schedule_attention_pipeline_with_chunks(flash_attention=False, inter_layer_chunk_size=inter_layer_chunk_size, sl_cross=sl_cross, atten_type=atten_type)
                elif pipelining:
                    sched, lat, chunk_size, mem, chunk_ready_times = self.schedule_attention_pipeline_with_chunks(flash_attention=False, sl_cross=sl_cross, atten_type=atten_type)
                elif inter_layer_pipelining:
                    sched, lat, chunk_size, mem, chunk_ready_times = self.schedule_attention_serial_with_chunks(
                        flash_attention=False,
                        inter_layer_pipelining=inter_layer_pipelining,
                        inter_layer_chunk_size=inter_layer_chunk_size,
                        sl_cross=sl_cross, atten_type=atten_type
                        )
                else:
                    ops = self.build_operation_graph_serial(
                        gemm_lat_QK, gemm_lat_QKV, softmax_lat,
                        dma_lat_Q, dma_lat_K, dma_lat_V, dma_out_lat,
                        flash_attention=False, sl_cross=sl_cross, atten_type=atten_type
                    )
                    sched, lat, mem = self.schedule_sequential_attention(ops)
                    chunk_ready_times = None
            else:
                print(f"Running Flash-Attention with warp size equal to {warp}")
                if pipelining and inter_layer_pipelining:
                    sched, lat, chunk_size, mem, chunk_ready_times = self.schedule_attention_pipeline_with_chunks(flash_attention=True, sl_warp=warp, inter_layer_chunk_size=inter_layer_chunk_size, sl_cross=sl_cross, atten_type=atten_type)        
                elif pipelining:
                    sched, lat, chunk_size, mem, chunk_ready_times = self.schedule_attention_pipeline_with_chunks(flash_attention=True, sl_warp=warp, sl_cross=sl_cross, atten_type=atten_type)        
                elif inter_layer_pipelining:
                    sched, lat, chunk_size, mem, chunk_ready_times = self.schedule_attention_serial_with_chunks(
                        flash_attention=True,
                        sl_warp=warp,
                        inter_layer_pipelining=inter_layer_pipelining,
                        inter_layer_chunk_size=inter_layer_chunk_size,
                        sl_cross=sl_cross, atten_type=atten_type
                        )
                else:
                    ops = self.build_operation_graph_serial(
                        gemm_lat_QK, gemm_lat_QKV, softmax_lat,
                        dma_lat_Q, dma_lat_K, dma_lat_V, dma_out_lat,
                        flash_attention=True,
                        sl_warp=warp, sl_cross=sl_cross, atten_type=atten_type
                    )
                    sched, lat, mem = self.schedule_sequential_attention(ops)
                    chunk_ready_times = None

            #sched, lat, mem = self.schedule_sequential_attention(ops)
            latencies[warp] = lat
            global_warp_latency_cache[cache_key] = (lat, sched, mem, chunk_ready_times)

            if verbose:
                print(f"  warp={warp:>3} (chunk size={chunk_size})  --> latency = {lat:.3f} ms")
            return lat, sched, mem, chunk_ready_times
        
        ### Avoid fine-grained autotune when chunk optimization is on for runtime purposes
        if (pipelining and self.optimize_attention_chunk) or inter_layer_pipelining:
            warp = self.sl_full if atten_type != "cross" else sl_cross

            best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times = None, float('inf'), None, None, None
            iterations = 0
            while warp >= 16:
                lat, sched, mem, chunk_ready_times = measure(warp, sl_to_use)
                if lat <= best_latency:
                    best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times= warp, lat, sched, mem, chunk_ready_times
                iterations += 1
                if iterations > 1 and lat!=float('inf'):
                    break
                warp = warp // 2

            if best_warp is None:
                raise RuntimeError("No valid sl_warp fits in TCDM for chunked FlashAttention.")

            if plot_curve:
                import matplotlib.pyplot as plt
                warp_vals = sorted(latencies.keys())
                lat_vals = [latencies[w] for w in warp_vals]
                plt.figure(figsize=(10, 4))
                plt.plot(warp_vals, lat_vals, marker='o')
                plt.axvline(best_warp, color='red', linestyle='--', label=f"Best warp: {best_warp}")
                plt.xlabel("sl_warp")
                plt.ylabel("Latency (ms)")
                plt.title("Latency vs sl_warp (FlashAttention Auto-Tune)")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"Plots/Attention/{self.model_name}_flash_attention_tuning.pdf")
                plt.show()

            self.write_flash_attn_tuning_log(
                model_name=self.model_name,
                sl_full=self.sl_full,
                latencies=latencies,
                tcdm_skipped=tcdm_skipped,
                used_flash=(best_warp != self.sl_full),
                best_warp=best_warp,
                best_latency=best_latency,
                best_chunk_ready_times=best_chunk_ready_times,
                pipelined=pipelining,
                parallelism=parallelism
            )

            return best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times
        # --- Adaptive granularity ---
        #granularity = max(16, int(2 ** (3 + self.sl_full // 128)))
        granularity = max(16, (sl_to_use // 8))
        coarse_step = granularity * 2  

        # --- Coarse search ---
        candidates = sorted(set(range(16, self.sl_full + 1, coarse_step)) | {self.sl_full})
        best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times= None, float('inf'), None, None, None

        for warp in candidates:
            lat, sched, mem, chunk_ready_times = measure(warp, sl_to_use)
            if lat <= best_latency:
                best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times = warp, lat, sched, mem, chunk_ready_times

        # --- Fine search around best candidate ---
        fine_range = range(best_warp - coarse_step, best_warp + coarse_step + 1, granularity)
        fine_candidates = [w for w in fine_range if 16 <= w <= self.sl_full and w not in visited]

        for warp in fine_candidates:
            lat, sched, mem, chunk_ready_times = measure(warp, sl_to_use)
            if lat <= best_latency:
                best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times = warp, lat, sched, mem, chunk_ready_times


        # --- Optional plot ---
        if plot_curve:
            import matplotlib.pyplot as plt
            warp_vals = sorted(latencies.keys())
            lat_vals = [latencies[w] for w in warp_vals]
            plt.figure(figsize=(10, 4))
            plt.plot(warp_vals, lat_vals, marker='o')
            plt.axvline(best_warp, color='red', linestyle='--', label=f"Best warp: {best_warp}")
            plt.xlabel("sl_warp")
            plt.ylabel("Latency (ms)")
            plt.title("Latency vs sl_warp (FlashAttention Auto-Tune)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"Plots/Attention/{self.model_name}_flash_attention_tuning.pdf")
            plt.show()

        # --- Logging ---
        self.write_flash_attn_tuning_log(
            model_name=self.model_name,
            sl_full=self.sl_full,
            latencies=latencies,
            tcdm_skipped=tcdm_skipped,
            used_flash=(best_warp != self.sl_full),
            best_warp=best_warp,
            best_latency=best_latency,
            best_chunk_ready_times=best_chunk_ready_times,
            pipelined=pipelining,
            parallelism=parallelism
        )

        return best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times
    
    @staticmethod
    def write_flash_attn_tuning_log(model_name, sl_full, latencies, tcdm_skipped, used_flash, best_warp, best_chunk_ready_times, best_latency, pipelined=False, parallelism=False):
        """
        Writes a detailed log of the FlashAttention warp tuning process.
        """
        log_path = Path(f"outputs/Attention/{model_name}_flash_attn_tuning_{pipelined}_pipe_{parallelism}_parall_sl{sl_full}.txt")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as f:
            f.write(f"# FlashAttention Warp Size Auto-Tuning Log\n")
            f.write(f"# Model: {model_name}\n")
            f.write(f"# Sequence Length (sl_full): {sl_full}\n")
            f.write(f"# Tuning Mode: {'Pipelined' if pipelined else 'Serial'}\n")
            f.write(f"# Timestamp: {datetime.now()}\n\n")

            f.write("## Candidate Warp Sizes\n")
            f.write("Warp Size | Latency (ms) | TCDM Fit | Notes\n")
            f.write("----------|--------------|-----------|------\n")
            for warp in sorted(latencies.keys()):
                latency = latencies[warp]
                fit = "YES" if warp not in tcdm_skipped else "NO"
                notes = ""
                if warp == best_warp:
                    notes = "<-- SELECTED"
                f.write(f"{warp:9d} | {latency:12.3f} | {fit:9s} | {notes}\n")

            f.write("\n## Final Decision\n")
            f.write(f"Best Warp Size     : {best_warp}\n")
            f.write(f"Latency Achieved   : {best_latency:.3f} ms\n")
            f.write(f"Used FlashAttention: {'YES' if used_flash else 'NO'}\n")
            f.write(f"Pipelined Mode     : {'YES' if pipelined else 'NO'}\n")

            if best_chunk_ready_times is not None:
                f.write("\n## Best Chunk Ready Times\n")
                f.write("Chunk ID | Ready Time (ms)\n")
                f.write("---------|-----------------\n")
                for chunk_id in sorted(best_chunk_ready_times.keys()):
                    f.write(f"{chunk_id:8d} | {best_chunk_ready_times[chunk_id]:.3f}\n")



    def total_latency(self, 
                      intra_layer_pipelining=None, 
                      intra_layer_parallelism=None,
                      inter_layer_pipelining=False,
                      flash_attention=False, 
                      sl_warp=None, 
                      auto_tune_sl_warp=False, 
                      inter_layer_chunk_size=None,
                      sl_cross = None,
                      atten_type = None,
                      plot=False, 
                      plot_memory=False
                      ):
        if atten_type == "cross":
            KV_size = sl_cross if (self.autoregressive) else self.sl_full
        else:
            KV_size = self.prefill_size if self.autoregressive else self.sl_full
        # softmax_lat = self.estimate_softmax_latency(self.sl,self.sl_full)
        # gemm_lat_QK = self.estimate_gemm_latency(self.sl,self.sl_full,self.head_dim)
        # gemm_lat_QKV = self.estimate_gemm_latency(self.sl,self.head_dim,self.sl_full)
        # dma_lat_Q = self.estimate_dma_latency(self.sl,self.head_dim)
        # dma_lat_K = self.estimate_dma_latency(self.sl_full,self.head_dim)
        # dma_lat_V = self.estimate_dma_latency(self.sl_full,self.head_dim)
        # dma_lat_OUT = self.estimate_dma_latency(self.sl,self.head_dim)
        softmax_lat = self.estimate_softmax_latency(self.sl,KV_size)
        gemm_lat_QK = self.estimate_gemm_latency(self.sl,KV_size,self.head_dim)
        gemm_lat_QKV = self.estimate_gemm_latency(self.sl,self.head_dim,KV_size)
        dma_lat_Q = self.estimate_dma_latency(self.sl,self.head_dim)
        dma_lat_K = self.estimate_dma_latency(KV_size,self.head_dim)
        dma_lat_V = self.estimate_dma_latency(KV_size,self.head_dim)
        dma_lat_OUT = self.estimate_dma_latency(self.sl,self.head_dim)

        # Handle FlashAttention + Auto-tune
        if flash_attention and auto_tune_sl_warp:
            self.sl_warp, best_latency, scheduled_ops, memory_trace, chunk_ready_times = self.tune_best_sl_warp(
                gemm_lat_QK, gemm_lat_QKV, softmax_lat,
                dma_lat_Q, dma_lat_K, dma_lat_V, dma_lat_OUT,
                plot_curve=plot,
                verbose=True,
                pipelining=intra_layer_pipelining,
                parallelism=intra_layer_parallelism,
                inter_layer_pipelining=inter_layer_pipelining, 
                inter_layer_chunk_size=inter_layer_chunk_size,
                atten_type=atten_type,
                sl_cross=sl_cross
            )
            print(f"[Auto-Tune] Selected sl_warp = {self.sl_warp}")
            if plot and not self.plot_once:
                AttentionPlotter.plot_gantt(scheduled_ops, intra_layer_pipelining, intra_layer_parallelism, flash_attention, self.model_name)
                if plot_memory:
                    AttentionPlotter.plot_memory_usage(memory_trace, intra_layer_pipelining, intra_layer_parallelism,flash_attention, self.model_name)
                self.plot_once = True
            return best_latency, chunk_ready_times

        # FlashAttention case with fixed sl_warp
        if flash_attention and KV_size > sl_warp:
            self.sl_warp = sl_warp
            if intra_layer_pipelining and inter_layer_pipelining:
                scheduled_ops, best_latency, best_chunk_size, memory_trace, chunk_ready_times = self.schedule_attention_pipeline_with_chunks(
                flash_attention=True, 
                sl_warp=self.sl_warp,
                inter_layer_chunk_size=inter_layer_chunk_size,
                sl_cross=sl_cross, 
                atten_type=atten_type
                )
                if plot and not self.plot_once:
                    AttentionPlotter.plot_gantt(scheduled_ops, intra_layer_pipelining, intra_layer_parallelism, flash_attention, self.model_name)
                    if plot_memory:
                        AttentionPlotter.plot_memory_usage(memory_trace, intra_layer_pipelining, intra_layer_parallelism, flash_attention,self.model_name)
                    self.plot_once = True
                return best_latency, chunk_ready_times
            elif intra_layer_pipelining :
                scheduled_ops, best_latency, best_chunk_size, memory_trace, chunk_ready_times = self.schedule_attention_pipeline_with_chunks(
                flash_attention=True, 
                sl_warp=self.sl_warp,
                sl_cross=sl_cross, 
                atten_type=atten_type
                )
                if plot and not self.plot_once:
                    AttentionPlotter.plot_gantt(scheduled_ops, intra_layer_pipelining, intra_layer_parallelism, flash_attention, self.model_name)
                    if plot_memory:
                        AttentionPlotter.plot_memory_usage(memory_trace, intra_layer_pipelining, intra_layer_parallelism, flash_attention,self.model_name)
                    self.plot_once = True
                return best_latency, chunk_ready_times
            elif inter_layer_pipelining and not intra_layer_pipelining:
                scheduled_ops, latency, _, memory_trace, chunk_ready_times = self.schedule_attention_serial_with_chunks(
                    flash_attention=True, sl_warp=sl_warp, inter_layer_pipelining=inter_layer_pipelining, inter_layer_chunk_size=inter_layer_chunk_size, sl_cross=sl_cross, atten_type=atten_type
                )
                if plot and not self.plot_once:
                    AttentionPlotter.plot_gantt(scheduled_ops, intra_layer_pipelining, intra_layer_parallelism, flash_attention, model_name=self.model_name)
                    if plot_memory:
                        AttentionPlotter.plot_memory_usage(memory_trace, intra_layer_pipelining, intra_layer_parallelism, flash_attention, self.model_name)
                    self.plot_once = True
                return latency, chunk_ready_times
            else:
                ops = self.build_operation_graph_serial(
                    gemm_lat_QK, gemm_lat_QKV, softmax_lat,
                    dma_lat_Q, dma_lat_K, dma_lat_V, dma_lat_OUT,
                    flash_attention=True,
                    sl_warp=self.sl_warp,
                    sl_cross=sl_cross,
                    atten_type=atten_type
                )
                scheduled_ops, latency, memory_trace = self.schedule_sequential_attention(ops)
                if plot and not self.plot_once:
                    AttentionPlotter.plot_gantt(scheduled_ops, intra_layer_pipelining, intra_layer_parallelism, flash_attention, model_name=self.model_name)
                    if plot_memory:
                        AttentionPlotter.plot_memory_usage(memory_trace, intra_layer_pipelining, intra_layer_parallelism,flash_attention, self.model_name)
                    self.plot_once = True
                return latency, latency
        else:
            if intra_layer_pipelining and inter_layer_pipelining:
                scheduled_ops, best_latency, best_chunk_size, memory_trace, chunk_ready_times = self.schedule_attention_pipeline_with_chunks(inter_layer_chunk_size=inter_layer_chunk_size, sl_cross=sl_cross, atten_type=atten_type)          
                if plot and not self.plot_once:
                    AttentionPlotter.plot_gantt(scheduled_ops, intra_layer_pipelining, intra_layer_parallelism, flash_attention, self.model_name)
                    if plot_memory:
                        AttentionPlotter.plot_memory_usage(memory_trace, intra_layer_pipelining, intra_layer_parallelism, flash_attention,self.model_name)
                    self.plot_once = True
                return best_latency, chunk_ready_times
            elif intra_layer_pipelining:
                scheduled_ops, best_latency, best_chunk_size, memory_trace, chunk_ready_times = self.schedule_attention_pipeline_with_chunks(sl_cross=sl_cross, atten_type=atten_type)          
                if plot and not self.plot_once:
                    AttentionPlotter.plot_gantt(scheduled_ops, intra_layer_pipelining, intra_layer_parallelism, flash_attention, self.model_name)
                    if plot_memory:
                        AttentionPlotter.plot_memory_usage(memory_trace, intra_layer_pipelining, intra_layer_parallelism, flash_attention,self.model_name)
                    self.plot_once = True
                return best_latency, chunk_ready_times
            elif inter_layer_pipelining:
                scheduled_ops, latency, _, memory_trace, chunk_ready_times = self.schedule_attention_serial_with_chunks(
                    inter_layer_pipelining=inter_layer_pipelining, inter_layer_chunk_size=inter_layer_chunk_size, sl_cross=sl_cross, atten_type=atten_type
                )
                if plot and not self.plot_once:
                    AttentionPlotter.plot_gantt(scheduled_ops, intra_layer_pipelining, intra_layer_parallelism, flash_attention, model_name=self.model_name)
                    if plot_memory:
                        AttentionPlotter.plot_memory_usage(memory_trace, intra_layer_pipelining, intra_layer_parallelism, flash_attention, self.model_name)
                    self.plot_once = True
                return latency, chunk_ready_times

            else:
                # Serial execution
                ops = self.build_operation_graph_serial(gemm_lat_QK, 
                                                        gemm_lat_QKV, 
                                                        softmax_lat, 
                                                        dma_lat_Q, 
                                                        dma_lat_K, 
                                                        dma_lat_V, 
                                                        dma_lat_OUT,
                                                        flash_attention=flash_attention, 
                                                        sl_warp=self.sl_warp if flash_attention else None,
                                                        sl_cross=sl_cross,
                                                        atten_type=atten_type)
                scheduled_ops, latency, memory_trace = self.schedule_sequential_attention(ops)
                if plot and not self.plot_once:
                    AttentionPlotter.plot_gantt(scheduled_ops, 
                                                intra_layer_pipelining, 
                                                intra_layer_parallelism, 
                                                flash_attention, 
                                                model_name=self.model_name)
                    if plot_memory:
                        AttentionPlotter.plot_memory_usage(memory_trace, intra_layer_pipelining, intra_layer_parallelism, flash_attention, self.model_name)
                    self.plot_once = True
                return latency, latency

class AttentionLatencyEstimatorDAs:
    def __init__(self, sl, head_dim, hidden_size, model_name="default", max_size_vect_parall=64, sl_full=None):
        self.sl = sl  # Chunked Q length
        self.sl_full = sl_full if sl_full is not None else sl  # Full K/V length
        self.head_dim = head_dim     # dimension per attention head
        self.hidden_size = hidden_size
        self.num_heads = hidden_size//head_dim
        self.model_name = model_name
        self.plot_once = 0
        self.chunk_latency_cache = {}
        self.optimize_attention_chunk = False
        self.max_size_vect_parall = max_size_vect_parall

        self.number_of_repetition = self.estimate_num_of_repetition(self.head_dim, self.num_heads)

    def estimate_dma_latency(self, M, N):
        """Estimate DMA transfer latency (ms)."""
        dma_bandwidth = DA0_TILE_CONFIG["dma_bandwidth"]
        size_B = M * N * (DA0_TILE_CONFIG["fp_precision"] / 8)
        latency = size_B/dma_bandwidth * DA0_TILE_CONFIG["clock"]
        return latency 
    
    def estimate_num_of_repetition(self, head_dim, num_heads):
        max_heads_in_parallel = DA0_TILE_CONFIG["num_cores"] / math.ceil(head_dim / self.max_size_vect_parall)
        num_of_repetition = math.ceil(num_heads/DA0_TILE_CONFIG["num_tiles"]/max_heads_in_parallel)
        return num_of_repetition

    def estimate_qk_latency(self, sl, sl_full, head_dim, number_of_repetition):
        """Estimate GEMM QKT latency (ms)."""
        num_dot_products = sl*sl_full
        num_qk_ops = (
        head_dim * number_of_repetition * num_dot_products / self.max_size_vect_parall
        )
        gemm_qk_cycles = 4 + num_qk_ops/sl - 1
        gemm_qk_latency = gemm_qk_cycles * DA0_TILE_CONFIG["clock"]
        return gemm_qk_latency 
    
    def estimate_softmax_latency(self, sl, sl_full):
        """ Estimate Softmax latency for MxN matrix (ms)."""
        num_softmax_ops = sl*sl_full
        softmax_latency_cycles = 3 + (math.ceil(num_softmax_ops / 32) - 1)
        return softmax_latency_cycles * DA0_TILE_CONFIG["clock"]
    
    def estimate_qkv_latency(self, sl, sl_full, head_dim, number_of_repetition):
        """Estimate GEMM QKV latency (ms)."""
        num_dot_products = sl*sl_full
        num_av_ops = head_dim * number_of_repetition * num_dot_products / self.max_size_vect_parall
        gemm_qkv_cycles = 1 + num_av_ops - 1
        gemm_qkv_latency = gemm_qkv_cycles * DA0_TILE_CONFIG["clock"]
        return gemm_qkv_latency 
    
    def build_operation_graph_serial(self, *latencies, flash_attention=False):
        """
        Build serial schedule for all heads.
        Latencies: GEMM_QK, GEMM_QKV, Softmax, DMA_Q, DMA_K
        """
        if flash_attention:
            raise RuntimeError("DA0 does not support FlashAttention!")
        gemm_qk, gemm_qkv, softmax_lat, dma_q_lat, dma_k_lat, dma_v_lat, dma_out_lat = latencies
        ops = []
        prev_dma_out = None

        for h in range(self.num_heads): 
            op = self._build_head_ops_serial(h, gemm_qk, gemm_qkv, softmax_lat, dma_q_lat, dma_k_lat, dma_v_lat, dma_out_lat,flash_attention)
            if prev_dma_out:
                op[0].add_dependency(prev_dma_out)
            ops.extend(op)
            prev_dma_out = op[-1]

        return ops
    
    def _build_head_ops_serial(self, head_idx, gemm_qk, gemm_qkv, softmax_lat, dma_q_lat, dma_k_lat, dma_v_lat, dma_out_lat, flash_attention=False):
        """Build a list of ops for a single head in serial."""
        q = f"Q_{head_idx}"
        
        compute_ops = []

        size = lambda m, n: m * n * DA0_TILE_CONFIG["fp_precision"] / 8 / 1024  # KB

        if not flash_attention:
            k = f"K_{head_idx}"
            v = f"V_{head_idx}"
            out_qk = f"OUT_QK_{head_idx}"
            out_qkv = f"OUT_QKV_{head_idx}"

            dma_qk = AttentionOps(f"DMA_QK_H{head_idx}", "DMA", 
                                  self.estimate_dma_latency(self.sl,self.head_dim) + self.estimate_dma_latency(self.sl_full,self.head_dim),
                                tcdm_alloc=[{"name": q, "size": size(self.sl, self.head_dim)},
                                            {"name": k, "size": size(self.sl_full, self.head_dim)}])

            gemm1 = AttentionOps(f"GEMM1_H{head_idx}", "GEMM", self.estimate_qk_latency(self.sl, self.sl_full, self.head_dim, self.number_of_repetition),
                                tcdm_alloc=[{"name": out_qk, "size": size(self.sl, self.sl_full)}],
                                tcdm_free=[q, k])

            softmax = AttentionOps(f"Softmax_H{head_idx}", "SOFTMAX", self.estimate_softmax_latency(self.sl, self.sl_full))

            dma_v = AttentionOps(f"DMA_V_H{head_idx}", "DMA", self.estimate_dma_latency(self.sl_full,self.head_dim),
                                tcdm_alloc=[{"name": v, "size": size(self.sl_full, self.head_dim)}])

            gemm2 = AttentionOps(f"GEMM2_H{head_idx}", "GEMM", self.estimate_qkv_latency(self.sl, self.sl_full, self.head_dim, self.number_of_repetition),
                                tcdm_alloc=[{"name": out_qkv, "size": size(self.sl, self.head_dim)}],
                                tcdm_free=[v, out_qk])

            dma_out = AttentionOps(f"DMA_OUT_H{head_idx}", "DMA", self.estimate_dma_latency(self.sl,self.head_dim),
                                tcdm_free=[out_qkv])

            # Dependencies
            gemm1.add_dependency(dma_qk)
            softmax.add_dependency(gemm1)
            dma_v.add_dependency(softmax)
            gemm2.add_dependency(softmax)
            gemm2.add_dependency(dma_v)
            dma_out.add_dependency(gemm2)

            return [dma_qk, gemm1, softmax, dma_v, gemm2, dma_out]
        
        else:
            raise RuntimeError("DA0 does not support FlashAttention!")
        

    def build_attention_ops_for_chunk(self, chunk_size, flash_attention=False, sl_warp=None):
        ops, chunk_ops = [], []
        num_chunks = math.ceil(self.sl / chunk_size)
        KB = PMCA_RED_TILE_CONFIG["fp_precision"] / 8 / 1024  # Convert to KB
        prev_dma_qk_or_q = None
        prev_dma_v = None

        for chunk_idx in range(num_chunks):
            chunk_ops_this_chunk = []
            sl_chunk = chunk_size if chunk_idx < num_chunks - 1 else self.sl - chunk_size * (num_chunks - 1)

            for head_idx in range(1): #for head_idx in range(self.num_heads):
                q_name = f"Q_chunk{chunk_idx}_H{head_idx}"
                size_q = sl_chunk * self.head_dim * KB

                if not flash_attention:
                    # --- Standard Attention: Q+K together ---
                    k_name = f"K_H{head_idx}"
                    v_name = f"V_H{head_idx}"
                    out_qk = f"OUT_QK_chunk{chunk_idx}_H{head_idx}"
                    out_qkv = f"OUT_QKV_chunk{chunk_idx}_H{head_idx}"

                    dma_qk = AttentionOps(
                        f"DMA_QK_chunk{chunk_idx}_H{head_idx}", "DMA",
                        duration= (self.estimate_dma_latency(sl_chunk, self.head_dim) + self.estimate_dma_latency(self.sl_full, self.head_dim)) if ((chunk_size==1 and chunk_idx==0) or chunk_size!=1) else 0,
                        tcdm_alloc=[{"name": q_name, "size": size_q},
                                    {"name": k_name, "size": self.sl_full * self.head_dim * KB}],
                        tcdm_free=[]
                    )

                    # Head-wise dependency
                    if head_idx == 0:
                        pass
                    elif head_idx % 2 == 1:
                        dma_qk.add_dependency(prev_dma_qk_or_q)
                    else :
                        dma_qk.add_dependency(prev_dma_v)

                    gemm1 = AttentionOps( 
                        f"GEMM1_chunk{chunk_idx}_H{head_idx}", "GEMM",
                        duration= (self.estimate_qk_latency(sl_chunk, self.sl_full, self.number_of_repetition, self.head_dim)) if ((chunk_size==1 and chunk_idx==0) or chunk_size!=1) else 0,
                        tcdm_alloc=[{"name": out_qk, "size": sl_chunk * self.sl_full * KB}],
                        tcdm_free=[q_name, k_name]
                    )

                    softmax = AttentionOps(
                        f"Softmax_chunk{chunk_idx}_H{head_idx}", "SOFTMAX",
                        duration= (self.estimate_softmax_latency(sl_chunk, self.sl_full)) if ((chunk_size==1 and chunk_idx==0) or chunk_size!=1) else 0,
                    )

                    dma_v = AttentionOps(
                        f"DMA_V_chunk{chunk_idx}_H{head_idx}", "DMA",
                        duration= (self.estimate_dma_latency(self.sl_full, self.head_dim)) if ((chunk_size==1 and chunk_idx==0) or chunk_size!=1) else 0,
                        tcdm_alloc=[{"name": v_name, "size": self.sl_full * self.head_dim * KB}],
                        tcdm_free=[]
                    )

                    gemm2 = AttentionOps(
                        f"GEMM2_chunk{chunk_idx}_H{head_idx}", "GEMM",
                        duration= (self.estimate_qkv_latency(sl_chunk, self.sl_full, self.number_of_repetition, self.head_dim)) if ((chunk_size==1 and chunk_idx==0) or chunk_size!=1) else 1*DA0_TILE_CONFIG["clock"],
                        tcdm_alloc=[{"name": out_qkv, "size": sl_chunk * self.head_dim * KB}],
                        tcdm_free=[v_name, out_qk]
                    )

                    dma_out = AttentionOps(
                        f"DMA_OUT_chunk{chunk_idx}_H{head_idx}", "DMA",
                        duration= (self.estimate_dma_latency(sl_chunk, self.head_dim)) if ((chunk_size==1 and chunk_idx==0) or chunk_size!=1) else 0,
                        tcdm_free=[out_qkv]
                    )

                    gemm1.add_dependency(dma_qk)
                    softmax.add_dependency(gemm1)
                    dma_v.add_dependency(gemm1)
                    gemm2.add_dependency(softmax)
                    gemm2.add_dependency(dma_v)
                    dma_out.add_dependency(gemm2)

                    chunk_ops_this_chunk.extend([dma_qk, gemm1, softmax, dma_v, gemm2, dma_out])
                    prev_dma_qk_or_q = dma_qk
                    prev_dma_v = dma_v

                else:
                    raise RuntimeError("DA0 does not support FlashAttention!")

            # --- Set dependencies across chunks ---
            if chunk_ops:
                num_heads = self.num_heads
                
                if self.number_of_repetition == 1:
                    if flash_attention:
                        prev_dma_qk_ops = [op for op in reversed(chunk_ops) if op.name.startswith("DMA_K")]
                    else:
                        prev_dma_qk_ops = [op for op in reversed(chunk_ops) if op.name.startswith("DMA_QK")]

                    prev_dma_v_ops  = [op for op in reversed(chunk_ops) if op.name.startswith("DMA_V")]

                    if chunk_idx % 2 == 0:
                        # Even chunk index: depend on last DMA_V
                        last_dma_v = prev_dma_v_ops[0]
                        for op in chunk_ops_this_chunk:
                            if op.name.startswith("DMA_QK") or op.name.startswith("DMA_Q") :
                                op.add_dependency(last_dma_v)
                    else:
                        # Odd chunk index: depend on last DMA_V
                        last_dma_qk = prev_dma_qk_ops[0]
                        for op in chunk_ops_this_chunk:
                            if op.name.startswith("DMA_QK") or op.name.startswith("DMA_Q") :
                                op.add_dependency(last_dma_qk)
                else:
                    # Multiple heads => must wait for second-to-last DMA_OUT
                    dma_out_ops = [op for op in chunk_ops if op.name.startswith("DMA_OUT")]
                    assert len(dma_out_ops) >= 2, "Not enough DMA_OUT ops to find second-to-last!"
                    second_last_dma_out = dma_out_ops[-2]
                    for op in chunk_ops_this_chunk:
                        if op.name.startswith("DMA_QK")  or op.name.startswith("DMA_Q") :
                            op.add_dependency(second_last_dma_out)

            ops.extend(chunk_ops_this_chunk)
            chunk_ops.extend(chunk_ops_this_chunk)

        return ops

    def schedule_sequential_attention(self, ops):
        current_time = 0.0
        scheduled_ops = []
        tcdm_capacity = DA0_TILE_CONFIG["tcdm_capacity"]
        tcdm_usage = []
        memory_trace = []

        log_filename = f"outputs/Attention/{self.model_name}_attention_tcdm_sequential_schedule_debug.txt"
        log_file = open(log_filename, "w")

        def free_expired_memory(now):
            expired = [mem for mem in tcdm_usage if (mem["free_time"] is not None) and (mem["free_time"] <= now)]
            if expired:
                print(f"[{now:.3f} ms] \u27a1\ufe0f  Freeing memory:", file=log_file)
                for mem in expired:
                    print(f"    - {mem['name']:10s} ({mem['size']:.2f} KB)", file=log_file)
            tcdm_usage[:] = [mem for mem in tcdm_usage if (mem["free_time"] is None) or (mem["free_time"] > now)]

        def available_tcdm():
            return tcdm_capacity - sum(mem["size"] for mem in tcdm_usage)

        for op in ops:
            while True:
                free_expired_memory(current_time)

                required_mem = sum(alloc["size"] for alloc in op.tcdm_alloc)
                if required_mem <= available_tcdm() and op.is_ready(current_time):
                    break  # We can schedule now

                # Check for next unlock event
                future_times = [d.end_time for d in op.dependencies if d.end_time > current_time]
                future_frees = [mem["free_time"] for mem in tcdm_usage if mem["free_time"] is not None and mem["free_time"] > current_time]
                next_event = min(future_times + future_frees, default=None)

                if next_event is None:
                    raise RuntimeError(f"Deadlock: cannot schedule {op.name} and no future events exist.")
                current_time = next_event

            # --- Schedule operation ---
            op.start_time = current_time
            op.end_time = current_time + op.duration
            scheduled_ops.append(op)

            print(f"[{current_time:.3f} ms] Scheduled {op.name:15s} on {op.op_type:8s} @ {op.start_time:.3f} dur={op.duration:.3f}", file=log_file)

            # --- Allocate memory
            for alloc in op.tcdm_alloc:
                tcdm_usage.append({
                    "name": alloc["name"],
                    "size": alloc["size"],
                    "free_time": None
                })

            for entry in tcdm_usage:
                if entry["name"] in op.tcdm_free:
                    entry["free_time"] = op.end_time

            current_time = op.end_time
            memory_trace.append((current_time, sum(mem["size"] for mem in tcdm_usage)))

        free_expired_memory(current_time)
        log_file.close()
        return scheduled_ops, max(op.end_time for op in scheduled_ops), memory_trace

    def schedule_attention_serial_with_chunks(self, flash_attention=False, sl_warp=None, inter_layer_pipelining=False, inter_layer_chunk_size=None):
        """
        Serial version of attention scheduling with chunks.
        Used when inter-layer pipelining is enabled but not intra-layer pipelining.
        """
        if inter_layer_pipelining:
            chunk_size = inter_layer_chunk_size
        else: 
            chunk_size = self.sl
        
        ops = self.build_attention_ops_for_chunk(chunk_size, flash_attention, sl_warp)
        scheduled_ops, total_latency, memory_trace = self.schedule_sequential_attention(ops)
        chunk_ready_times = self._extract_chunk_ready_times(scheduled_ops)
        return scheduled_ops, total_latency, chunk_size, memory_trace, chunk_ready_times

    def schedule_attention_pipeline(self, ops, skip_inf_lat=False):
        current_time = 0.0
        scheduled_ops = []
        state = {
            "da_busy_until": -1,
            "dma_busy_until": -1,
        }
        tcdm_capacity = DA0_TILE_CONFIG["tcdm_capacity"]
        tcdm_usage = []
        memory_trace = []
        running_ops = []
        rollback_window = 3

        log_file = open(f"outputs/Attention/{self.model_name}_attention_tcdm_pipeline_schedule_debug.txt", "w")

        max_idle_time = 5.0
        idle_time = 0.0

        # -------------------------------------------------------------------
        # keep track of every "snapshot" of scheduled_ops (by name order)
        seen_schedules = []  # list of tuples, each tuple is (op.name, op.name, ...)
        seen_set = set()     # a set of those tuples for quick membership check
        # -------------------------------------------------------------------

        def free_expired_memory(now):
            expired = [mem for mem in tcdm_usage if mem["free_time"] is not None and mem["free_time"] <= now]
            for mem in expired:
                print(f"[{now:.3f} ms] \u27a1\ufe0f  Freeing {mem['name']:10s} ({mem['size']:.2f} KB)", file=log_file)
            tcdm_usage[:] = [mem for mem in tcdm_usage if mem["free_time"] is None or mem["free_time"] > now]

        def available_tcdm():
            return tcdm_capacity - sum(mem["size"] for mem in tcdm_usage)

        def score_op_priority(op):
            alloc = sum(a["size"] for a in op.tcdm_alloc)
            frees = sum(mem["size"] for mem in tcdm_usage if mem["name"] in op.tcdm_free)
            if alloc > available_tcdm():
                return float('-inf')

            # Extract chunk and head index from op name
            chunk_id, head_id, warp_id = 999, 999, 999
            if "chunk" in op.name and "_H" in op.name:
                try:
                    chunk_part = re.search(r"chunk(\d+)", op.name)
                    head_part = re.search(r"_H(\d+)", op.name)
                    warp_part = re.search(r"_w(\d+)", op.name)

                    if chunk_part:
                        chunk_id = int(chunk_part.group(1))
                    if head_part:
                        head_id = int(head_part.group(1))
                    if warp_part:
                        warp_id = int(warp_part.group(1))
                except Exception:
                    pass

            # Prioritize freeing memory
            score = frees - alloc

            # Strong priority: lower chunk, then head, then warp
            score += (10000 - 1000 * chunk_id - 100 * head_id - warp_id)

            # Slight bonus to GEMM2 to free buffers sooner
            if "GEMM2" in op.name:
                score += 10

            return score


        def get_ready_ops(current_time):
            return [op for op in ops if op.start_time is None and op.is_ready(current_time)]

        def rollback_and_reschedule():
            for i in range(1, rollback_window + 1):
                if len(scheduled_ops) < i:
                    break

                rollback_op = scheduled_ops[-1] # scheduled_ops[-i]
                print(f"[{current_time:.3f} ms] \U0001f504 Rolling back {rollback_op.name}", file=log_file)
                scheduled_ops.remove(rollback_op)
                rollback_op.start_time = None
                rollback_op.end_time = None
                tcdm_usage[:] = [mem for mem in tcdm_usage if mem["name"] not in [a["name"] for a in rollback_op.tcdm_alloc]]

                ready_now = get_ready_ops(current_time)
                ready_now = [op for op in ready_now if op != rollback_op]
                ready_now.sort(key=score_op_priority, reverse=True)

                for candidate in ready_now:
                    if candidate.op_type == "DMA" and current_time < state["dma_busy_until"]:
                        continue
                    if candidate.op_type == "SOFTMAX" and current_time < state["da_busy_until"]:
                        continue
                    if candidate.op_type == "GEMM" and current_time < state["da_busy_until"]:
                        continue
                    if sum(a["size"] for a in candidate.tcdm_alloc) > available_tcdm():
                        continue

                    candidate.start_time = current_time
                    candidate.end_time = current_time + candidate.duration
                    scheduled_ops.append(candidate)
                    if candidate.op_type == "GEMM":
                        state["da_busy_until"] = candidate.end_time
                    if candidate.op_type == "DMA":
                        state["dma_busy_until"] = candidate.end_time
                    if candidate.op_type == "SOFTMAX":
                        state["da_busy_until"] = candidate.end_time
                    for alloc in candidate.tcdm_alloc:
                        print(f"[{current_time:.3f} ms] \u27a1\ufe0f  Realloc {alloc['name']:10s} ({alloc['size']:.2f} KB)", file=log_file)
                        tcdm_usage.append({
                            "name": alloc["name"],
                            "size": alloc["size"],
                            "free_time": None
                        })
                    for entry in tcdm_usage:
                        if entry["name"] in candidate.tcdm_free:
                            entry["free_time"] = candidate.end_time
                    print(f"[{current_time:.3f} ms] \u2705 Rescheduled {candidate.name}", file=log_file)
                    return True

            return False

        while True:
            free_expired_memory(current_time)
            ready_ops = get_ready_ops(current_time)
            ready_ops.sort(key=score_op_priority, reverse=True)

            something_scheduled = False
            for op in ready_ops:
                if op.op_type == "GEMM" and current_time < state["da_busy_until"]:
                        continue
                if op.op_type == "DMA" and current_time < state["dma_busy_until"]:
                    continue
                if op.op_type == "SOFTMAX" and current_time < state["da_busy_until"]:
                    continue
                if sum(alloc["size"] for alloc in op.tcdm_alloc) > available_tcdm():
                    continue

                op.start_time = current_time
                op.end_time = current_time + op.duration
                scheduled_ops.append(op)

                if op.op_type == "GEMM":
                    state["da_busy_until"] = op.end_time
                if op.op_type == "DMA":
                    state["dma_busy_until"] = op.end_time
                if op.op_type == "SOFTMAX":
                    state["da_busy_until"] = op.end_time

                print(f"[{current_time:.3f} ms] Scheduled {op.name:15s} on {op.op_type:8s} @ {op.start_time:.3f} dur={op.duration:.3f}", file=log_file)
                for alloc in op.tcdm_alloc:
                    print(f"[{current_time:.3f} ms] \u27a1\ufe0f  Allocating {alloc['name']:10s} ({alloc['size']:.2f} KB)", file=log_file)
                    tcdm_usage.append({
                        "name": alloc["name"],
                        "size": alloc["size"],
                        "free_time": None
                    })
                for mem in tcdm_usage:
                    if mem["name"] in op.tcdm_free:
                        mem["free_time"] = op.end_time

                something_scheduled = True
                break

            memory_trace.append((current_time, sum(m["size"] for m in tcdm_usage)))

            if not get_ready_ops(current_time) and all(op.end_time <= current_time for op in scheduled_ops):
                break

            if not something_scheduled:
                next_events = [
                    t for t in [state["da_busy_until"], state["dma_busy_until"]] + 
                    [op.end_time for op in scheduled_ops if op.end_time > current_time] + 
                    [mem["free_time"] for mem in tcdm_usage if mem["free_time"] and mem["free_time"] > current_time]
                ]
                next_events = [t for t in next_events if t > current_time]

                if next_events:
                    current_time = min(next_events)
                    idle_time = 0.0
                else:
                    current_sequence = tuple(op.name for op in scheduled_ops)
                    if current_sequence in seen_set:
                        if not skip_inf_lat:
                            print("Deadlock: repeated scheduling order", file=log_file)
                            waiting_ops = [op.name for op in get_ready_ops(current_time)]
                            print(f"[{current_time:.3f} ms] Waiting ops: {waiting_ops}", file=log_file)
                            log_file.close()
                            return scheduled_ops, float("inf"), memory_trace
                        else:
                            raise RuntimeError("Deadlock: repeated scheduling order")
                    # 3) otherwise, record it and continue with rollback
                    seen_set.add(current_sequence)
                    seen_schedules.append(current_sequence)

                    print(">>> Attempting rollback due to deadlock", file=log_file)
                    if not rollback_and_reschedule():
                        if not skip_inf_lat:
                            print("Deadlock: rollback failed to resolve.", file=log_file)
                            waiting_ops = [op.name for op in get_ready_ops(current_time)]
                            print(f"[{current_time:.3f} ms] Waiting ops: {waiting_ops}", file=log_file)
                            log_file.close()
                            return scheduled_ops, float("inf"), memory_trace
                        else:
                            raise RuntimeError("Deadlock: rollback failed to resolve.")
                        
            else:
                idle_time = 0.0
                

            if idle_time > max_idle_time:
                raise RuntimeError(f"Deadlock detected after {idle_time:.3f} ms idle!")

        log_file.close()
        return scheduled_ops, max(op.end_time for op in scheduled_ops), memory_trace

    def schedule_attention_pipeline_with_chunks(self, flash_attention=False, sl_warp=None, inter_layer_chunk_size=None):
        """
        Try multiple chunk sizes and return the one with lowest latency.
        """
            # --- Fast path: use fixed chunk size from inter-layer pipelining --- can be merged with next if , if inter_layer_chunk_size is sent=SL when inter_layer pipeline is False
        if not self.optimize_attention_chunk and inter_layer_chunk_size is not None:
            ops = self.build_attention_ops_for_chunk(inter_layer_chunk_size, flash_attention, sl_warp)
            scheduled_ops, total_latency, memory_trace = self.schedule_attention_pipeline(ops)
            chunk_ready_times = self._extract_chunk_ready_times(scheduled_ops)
            return scheduled_ops, total_latency, inter_layer_chunk_size, memory_trace, chunk_ready_times

        if not self.optimize_attention_chunk:
            # Quick path: No optimization, use SL as one chunk
            ops = self.build_attention_ops_for_chunk(self.sl, flash_attention, sl_warp)
            scheduled_ops, total_latency, memory_trace = self.schedule_attention_pipeline(ops)
            chunk_ready_times = self._extract_chunk_ready_times(scheduled_ops)

            if flash_attention:
                log_msg = f" sl_warp {sl_warp}: total latency = {total_latency:.3f} ms"
            else:
                log_msg = f"total latency = {total_latency:.3f} ms"
            print(log_msg)
            return scheduled_ops, total_latency, self.sl, memory_trace, chunk_ready_times

        chunk_candidates = self.generate_chunk_candidates(self.sl)

        best = (None, float('inf'), None, None)

        cache_key = (self.sl, self.head_dim, self.hidden_size, flash_attention, sl_warp)
        global global_chunk_latency_cache

        log_filename = f"outputs/Attention/{self.model_name}_attention_tcdm_schedule_chunked_debug{'_FA' if flash_attention==True else ''}{'_inter_layer_chunk' if inter_layer_chunk_size != None else ''}{inter_layer_chunk_size if inter_layer_chunk_size != None else ''}.txt"
        log_file = open(log_filename, "a")

        print(f"[Chunk Optimization] Model: {self.model_name}, SL: {self.sl}, FlashAttention: {flash_attention}, SL_WARP: {sl_warp}", file=log_file)

        if  cache_key in global_chunk_latency_cache:
            print(f"[Cache] Using cached result for sl_warp={sl_warp}", file=log_file)
            return global_chunk_latency_cache[cache_key]

        for chunk_size in chunk_candidates:
            # Build ops for current chunk size
            ops = self.build_attention_ops_for_chunk(chunk_size,flash_attention, sl_warp)

            # Run normal scheduling (your existing code)
            sched, lat, mem = self.schedule_attention_pipeline(ops, skip_inf_lat=True)

            if flash_attention:
                log_msg = f"Chunk size {chunk_size}, sl_warp {sl_warp}: total latency = {lat:.3f} ms"
            else:
                log_msg = f"Chunk size {chunk_size}: total latency = {lat:.3f} ms"

            print(log_msg)
            print(log_msg, file=log_file)
            ###-------------------
            if lat == float("inf"):
                continue
            ###-------------------

            chunk_ready_times = self._extract_chunk_ready_times(sched)

            if lat < best[1]:
                best = (sched, lat, chunk_size, mem, chunk_ready_times)
            
        log_msg = f"[Best Chunk] Size = {best[2]}, latency = {best[1]:.3f} ms"
        print(log_msg)
        print(log_msg, file=log_file)
        

        global_chunk_latency_cache[cache_key] = best
        log_file.close()
        return best
    
    def _extract_chunk_ready_times(self, scheduled_ops):
        """
        Parse all scheduled ops and return the latest end time among all DMA_OUT_chunk* ops
        per chunk (across all heads).
        Returns: {chunk_idx: max_end_time_across_heads}.
        """
        import re
        from collections import defaultdict

        chunk_to_max_time = defaultdict(float)
        chunk_to_head_times = defaultdict(list)

        for op in scheduled_ops:
            if op.name.startswith("DMA_OUT_chunk"):
                match = re.search(r"chunk(\d+)_H(\d+)", op.name)
                if match:
                    chunk_id = int(match.group(1))
                    head_id = int(match.group(2))
                    chunk_to_head_times[chunk_id].append(op.end_time)

        for chunk_id, times in chunk_to_head_times.items():
            chunk_to_max_time[chunk_id] = max(times)

        return chunk_to_max_time


    def generate_chunk_candidates(self, sl):
        """
        Generate chunk sizes that are multiples of 16 and <= SL.
        """
        return [8] + [x for x in range(16, sl+1, 16)]

    def tune_best_sl_warp(self, gemm_lat_QK, gemm_lat_QKV, softmax_lat, 
                      dma_lat_Q, dma_lat_K, dma_lat_V, dma_out_lat, pipelining=False, parallelism=False,
                      inter_layer_pipelining=False, inter_layer_chunk_size=None,
                      plot_curve=False, verbose=False):
        """
        Automatically find the best sl_warp value to minimize attention latency.
        Uses a coarse-to-fine binary-like search, with:
        - Coarse search: large step (power-of-2)
        - Fine search: local refinement (smaller step)
        - TCDM-aware filtering of invalid warp sizes
        - Fair non-flash attention baseline when warp == sl_full
        - Early termination if fine search shows no improvement
        Returns: best_warp, best_latency, best_schedule, best_memory_trace
        """
        latencies = {}
        visited = set()
        tcdm_skipped = []
        chunk_ready_times = {}
        
        dummy_layer = type("Dummy", (), {})()
        dummy_layer.type = "atten"
        dummy_layer.input_shape = (self.sl, self.head_dim)
        dummy_layer.output_shape = (self.sl, self.hidden_size)
        dummy_layer.parent_layer = type("Parent", (), {"input_shape": (self.sl_full, self.head_dim)})
        
        # Cache to avoid recomputing latency
        global global_warp_latency_cache

        def measure(warp):
            if warp in visited or warp < 8 or warp > self.sl_full:
                return float('inf'), None, None
            visited.add(warp)

            chunk_size = None

            cache_key = (self.model_name, self.sl, self.head_dim, self.hidden_size, warp, pipelining, inter_layer_chunk_size)
            if cache_key in global_warp_latency_cache:
                if verbose:
                    print(f"[Cache] Using cached latency for warp={warp}")
                lat, sched, mem, chunk_ready_times = global_warp_latency_cache[cache_key]
                latencies[warp] = lat
                return lat, sched, mem, chunk_ready_times

            # --- Check TCDM memory constraint ---
            mem_kb = MemoryEstimator.estimate_memory_kb(dummy_layer, sl_full=warp, flash_attention=True, chunk_size=inter_layer_chunk_size if inter_layer_chunk_size!=None else self.sl)
            if mem_kb > DA0_TILE_CONFIG["tcdm_capacity"]:
                tcdm_skipped.append(warp)
                if verbose:
                    print(f"  warp={warp:>3}  --> skipped (memory {mem_kb:.1f} KB > TCDM)")
                return float('inf'), None, None, None

            # --- Fair baseline: no flash attention when warp == sl_full ---
            if warp == self.sl_full:
                print(f"Running Standard Attention with warp size equal to SL ({warp})")
                if pipelining and inter_layer_pipelining:
                    sched, lat, chunk_size, mem, chunk_ready_times = self.schedule_attention_pipeline_with_chunks(flash_attention=False, inter_layer_chunk_size=inter_layer_chunk_size)
                elif pipelining:
                    sched, lat, chunk_size, mem, chunk_ready_times = self.schedule_attention_pipeline_with_chunks(flash_attention=False)
                elif inter_layer_pipelining:
                    sched, lat, chunk_size, mem, chunk_ready_times = self.schedule_attention_serial_with_chunks(
                        flash_attention=False,
                        inter_layer_pipelining=inter_layer_pipelining,
                        inter_layer_chunk_size=inter_layer_chunk_size
                        )
                else:
                    ops = self.build_operation_graph_serial(
                        gemm_lat_QK, gemm_lat_QKV, softmax_lat,
                        dma_lat_Q, dma_lat_K, dma_lat_V, dma_out_lat,
                        flash_attention=False
                    )
                    sched, lat, mem = self.schedule_sequential_attention(ops)
                    chunk_ready_times = None
            else:
                print(f"Running Flash-Attention with warp size equal to {warp}")
                if pipelining and inter_layer_pipelining:
                    sched, lat, chunk_size, mem, chunk_ready_times = self.schedule_attention_pipeline_with_chunks(flash_attention=True, sl_warp=warp, inter_layer_chunk_size=inter_layer_chunk_size)        
                elif pipelining:
                    sched, lat, chunk_size, mem, chunk_ready_times = self.schedule_attention_pipeline_with_chunks(flash_attention=True, sl_warp=warp)        
                elif inter_layer_pipelining:
                    sched, lat, chunk_size, mem, chunk_ready_times = self.schedule_attention_serial_with_chunks(
                        flash_attention=True,
                        sl_warp=warp,
                        inter_layer_pipelining=inter_layer_pipelining,
                        inter_layer_chunk_size=inter_layer_chunk_size
                        )
                else:
                    ops = self.build_operation_graph_serial(
                        gemm_lat_QK, gemm_lat_QKV, softmax_lat,
                        dma_lat_Q, dma_lat_K, dma_lat_V, dma_out_lat,
                        flash_attention=True,
                        sl_warp=warp
                    )
                    sched, lat, mem = self.schedule_sequential_attention(ops)
                    chunk_ready_times = None

            #sched, lat, mem = self.schedule_sequential_attention(ops)
            latencies[warp] = lat
            global_warp_latency_cache[cache_key] = (lat, sched, mem, chunk_ready_times)

            if verbose:
                print(f"  warp={warp:>3} (chunk size={chunk_size})  --> latency = {lat:.3f} ms")
            return lat, sched, mem, chunk_ready_times
        
        ### Avoid fine-grained autotune when chunk optimization is on for runtime purposes
        if (pipelining and self.optimize_attention_chunk) or inter_layer_pipelining:
            warp = self.sl_full

            best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times = None, float('inf'), None, None, None
            iterations = 0
            while warp >= 16:
                lat, sched, mem, chunk_ready_times = measure(warp)
                if lat <= best_latency:
                    best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times= warp, lat, sched, mem, chunk_ready_times
                iterations += 1
                if iterations > 1 and lat!=float('inf'):
                    break
                warp = warp // 2

            if best_warp is None:
                raise RuntimeError("No valid sl_warp fits in TCDM for chunked FlashAttention.")

            if plot_curve:
                import matplotlib.pyplot as plt
                warp_vals = sorted(latencies.keys())
                lat_vals = [latencies[w] for w in warp_vals]
                plt.figure(figsize=(10, 4))
                plt.plot(warp_vals, lat_vals, marker='o')
                plt.axvline(best_warp, color='red', linestyle='--', label=f"Best warp: {best_warp}")
                plt.xlabel("sl_warp")
                plt.ylabel("Latency (ms)")
                plt.title("Latency vs sl_warp (FlashAttention Auto-Tune)")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"Plots/Attention/{self.model_name}_flash_attention_tuning.pdf")
                plt.show()

            self.write_flash_attn_tuning_log(
                model_name=self.model_name,
                sl_full=self.sl_full,
                latencies=latencies,
                tcdm_skipped=tcdm_skipped,
                used_flash=(best_warp != self.sl_full),
                best_warp=best_warp,
                best_latency=best_latency,
                best_chunk_ready_times=best_chunk_ready_times,
                pipelined=pipelining,
                parallelism=parallelism
            )

            return best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times
        # --- Adaptive granularity ---
        granularity = max(16, (self.sl_full // 8))
        coarse_step = granularity * 2  

        # --- Coarse search ---
        candidates = sorted(set(range(16, self.sl_full + 1, coarse_step)) | {self.sl_full})
        best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times= None, float('inf'), None, None, None

        for warp in candidates:
            lat, sched, mem, chunk_ready_times = measure(warp)
            if lat <= best_latency:
                best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times = warp, lat, sched, mem, chunk_ready_times

        # --- Fine search around best candidate ---
        fine_range = range(best_warp - coarse_step, best_warp + coarse_step + 1, granularity)
        fine_candidates = [w for w in fine_range if 16 <= w <= self.sl_full and w not in visited]

        for warp in fine_candidates:
            lat, sched, mem, chunk_ready_times = measure(warp)
            if lat <= best_latency:
                best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times = warp, lat, sched, mem, chunk_ready_times


        # --- Optional plot ---
        if plot_curve:
            import matplotlib.pyplot as plt
            warp_vals = sorted(latencies.keys())
            lat_vals = [latencies[w] for w in warp_vals]
            plt.figure(figsize=(10, 4))
            plt.plot(warp_vals, lat_vals, marker='o')
            plt.axvline(best_warp, color='red', linestyle='--', label=f"Best warp: {best_warp}")
            plt.xlabel("sl_warp")
            plt.ylabel("Latency (ms)")
            plt.title("Latency vs sl_warp (FlashAttention Auto-Tune)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"Plots/Attention/{self.model_name}_flash_attention_tuning.pdf")
            plt.show()

        # --- Logging ---
        self.write_flash_attn_tuning_log(
            model_name=self.model_name,
            sl_full=self.sl_full,
            latencies=latencies,
            tcdm_skipped=tcdm_skipped,
            used_flash=(best_warp != self.sl_full),
            best_warp=best_warp,
            best_latency=best_latency,
            best_chunk_ready_times=best_chunk_ready_times,
            pipelined=pipelining,
            parallelism=parallelism
        )

        return best_warp, best_latency, best_sched, best_mem, best_chunk_ready_times
    
    @staticmethod
    def write_flash_attn_tuning_log(model_name, sl_full, latencies, tcdm_skipped, used_flash, best_warp, best_chunk_ready_times, best_latency, pipelined=False, parallelism=False):
        """
        Writes a detailed log of the FlashAttention warp tuning process.
        """
        log_path = Path(f"outputs/Attention/{model_name}_flash_attn_tuning_{pipelined}_pipe_{parallelism}_parall_sl{sl_full}.txt")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as f:
            f.write(f"# FlashAttention Warp Size Auto-Tuning Log\n")
            f.write(f"# Model: {model_name}\n")
            f.write(f"# Sequence Length (sl_full): {sl_full}\n")
            f.write(f"# Tuning Mode: {'Pipelined' if pipelined else 'Serial'}\n")
            f.write(f"# Timestamp: {datetime.now()}\n\n")

            f.write("## Candidate Warp Sizes\n")
            f.write("Warp Size | Latency (ms) | TCDM Fit | Notes\n")
            f.write("----------|--------------|-----------|------\n")
            for warp in sorted(latencies.keys()):
                latency = latencies[warp]
                fit = "YES" if warp not in tcdm_skipped else "NO"
                notes = ""
                if warp == best_warp:
                    notes = "<-- SELECTED"
                f.write(f"{warp:9d} | {latency:12.3f} | {fit:9s} | {notes}\n")

            f.write("\n## Final Decision\n")
            f.write(f"Best Warp Size     : {best_warp}\n")
            f.write(f"Latency Achieved   : {best_latency:.3f} ms\n")
            f.write(f"Used FlashAttention: {'YES' if used_flash else 'NO'}\n")
            f.write(f"Pipelined Mode     : {'YES' if pipelined else 'NO'}\n")

            if best_chunk_ready_times is not None:
                f.write("\n## Best Chunk Ready Times\n")
                f.write("Chunk ID | Ready Time (ms)\n")
                f.write("---------|-----------------\n")
                for chunk_id in sorted(best_chunk_ready_times.keys()):
                    f.write(f"{chunk_id:8d} | {best_chunk_ready_times[chunk_id]:.3f}\n")



    def total_latency(self, 
                      intra_layer_pipelining=None, 
                      intra_layer_parallelism=None,
                      inter_layer_pipelining=False,
                      flash_attention=False, 
                      sl_warp=None, 
                      auto_tune_sl_warp=False, 
                      inter_layer_chunk_size=None,
                      plot=False, 
                      plot_memory=False
                      ):
        
        softmax_lat = self.estimate_softmax_latency(self.sl,self.sl_full)
        gemm_lat_QK = self.estimate_qk_latency(self.sl,self.sl_full, self.head_dim, self.number_of_repetition)
        gemm_lat_QKV = self.estimate_qkv_latency(self.sl,self.sl_full, self.head_dim, self.number_of_repetition)
        dma_lat_Q = self.estimate_dma_latency(self.sl,self.head_dim)
        dma_lat_K = self.estimate_dma_latency(self.sl_full,self.head_dim)
        dma_lat_V = self.estimate_dma_latency(self.sl_full,self.head_dim)
        dma_lat_OUT = self.estimate_dma_latency(self.sl,self.head_dim)

        # FlashAttention case with fixed sl_warp
        if flash_attention:
            raise ValueError("DA0 does not support FlashAttention.")
        else:
            if intra_layer_pipelining and inter_layer_pipelining:
                scheduled_ops, best_latency, best_chunk_size, memory_trace, chunk_ready_times = self.schedule_attention_pipeline_with_chunks(inter_layer_chunk_size=inter_layer_chunk_size)          
                if plot and not self.plot_once:
                    AttentionPlotter.plot_gantt(scheduled_ops, intra_layer_pipelining, intra_layer_parallelism, flash_attention, self.model_name)
                    if plot_memory:
                        AttentionPlotter.plot_memory_usage(memory_trace, intra_layer_pipelining, intra_layer_parallelism, flash_attention,self.model_name)
                    self.plot_once = True
                return best_latency, chunk_ready_times
            elif intra_layer_pipelining:
                scheduled_ops, best_latency, best_chunk_size, memory_trace, chunk_ready_times = self.schedule_attention_pipeline_with_chunks()          
                if plot and not self.plot_once:
                    AttentionPlotter.plot_gantt(scheduled_ops, intra_layer_pipelining, intra_layer_parallelism, flash_attention, self.model_name)
                    if plot_memory:
                        AttentionPlotter.plot_memory_usage(memory_trace, intra_layer_pipelining, intra_layer_parallelism, flash_attention,self.model_name)
                    self.plot_once = True
                return best_latency, chunk_ready_times
            elif inter_layer_pipelining:
                scheduled_ops, latency, _, memory_trace, chunk_ready_times = self.schedule_attention_serial_with_chunks(
                    inter_layer_pipelining=inter_layer_pipelining, inter_layer_chunk_size=inter_layer_chunk_size
                )
                if plot and not self.plot_once:
                    AttentionPlotter.plot_gantt(scheduled_ops, intra_layer_pipelining, intra_layer_parallelism, flash_attention, model_name=self.model_name)
                    if plot_memory:
                        AttentionPlotter.plot_memory_usage(memory_trace, intra_layer_pipelining, intra_layer_parallelism, flash_attention, self.model_name)
                    self.plot_once = True
                return latency, chunk_ready_times

            else:
                # Serial execution
                ops = self.build_operation_graph_serial(gemm_lat_QK, 
                                                        gemm_lat_QKV, 
                                                        softmax_lat, 
                                                        dma_lat_Q, 
                                                        dma_lat_K, 
                                                        dma_lat_V, 
                                                        dma_lat_OUT,
                                                        flash_attention=flash_attention)
                scheduled_ops, latency, memory_trace = self.schedule_sequential_attention(ops)
                if plot and not self.plot_once:
                    AttentionPlotter.plot_gantt(scheduled_ops, 
                                                intra_layer_pipelining, 
                                                intra_layer_parallelism, 
                                                flash_attention, 
                                                model_name=self.model_name)
                    if plot_memory:
                        AttentionPlotter.plot_memory_usage(memory_trace, intra_layer_pipelining, intra_layer_parallelism, flash_attention, self.model_name)
                    self.plot_once = True
                return latency, latency



class AttentionOps:
    def __init__(self, name, op_type, duration, tcdm_alloc=[], tcdm_free=[]):
        self.name = name
        self.op_type = op_type  # 'DMA', 'GEMM', 'SOFTMAX'
        self.duration = duration
        self.tcdm_alloc = tcdm_alloc  # list of {name, size}
        self.tcdm_free = tcdm_free 
        self.dependencies = []
        self.start_time = None
        self.end_time = None
        
    def add_dependency(self, op):
        self.dependencies.append(op)

    def is_ready(self, current_time):
        return all(dep.end_time is not None and dep.end_time <= current_time for dep in self.dependencies)

class AttentionPlotter:
    @staticmethod
    def plot_gantt(scheduled_ops, intra_layer_pipelining=False, intra_layer_parallelism=False, flash_attention=False, model_name="default"):
        
        fig, ax = plt.subplots(figsize=(12, 6))

        # Colors for different op types
        colors = {
            "DMA": "tab:blue",
            "GEMM": "tab:red",
            "SOFTMAX": "tab:green",
            "DA": "tab:purple",
        }

        yticks = []
        yticklabels = []
        for idx, op in enumerate(scheduled_ops):
            start = op.start_time
            duration = op.end_time - op.start_time
            color = colors.get(op.op_type, "gray")

            ax.barh(idx, duration, left=start, color=color, edgecolor="black")
            ax.text(start + duration/2, idx, op.name, va='center', ha='center', color="white", fontsize=8)

            yticks.append(idx)
            yticklabels.append(op.name)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Operations")
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        if intra_layer_pipelining:
            ax.set_title("Intra-Layer Pipelining (Attention)")
        else:
            ax.set_title("Serial operations (Attention)")

        intra_layer_pipelining_type = "intra_layer" if intra_layer_pipelining else "None"
        intra_layer_parallelism_type = "intra_layer" if intra_layer_parallelism else "None"

        # Create legend
        patches = [mpatches.Patch(color=colors[typ], label=typ) for typ in colors]
        ax.legend(handles=patches, loc='upper right')

        plt.grid(True)
        plt.tight_layout()
        if flash_attention:
            plt.savefig(f"Plots/Attention/{model_name}_att_scheduling_{intra_layer_pipelining_type}_pipe_and_{intra_layer_parallelism_type}_parall_FA.pdf")
            plt.savefig(f"Plots/Attention/{model_name}_att_scheduling_{intra_layer_pipelining_type}_pipe_and_{intra_layer_parallelism_type}_parall_FA.svg")
        else:
            plt.savefig(f"Plots/Attention/{model_name}_att_scheduling_{intra_layer_pipelining_type}_pipe_and_{intra_layer_parallelism_type}_parall.pdf")
            plt.savefig(f"Plots/Attention/{model_name}_att_scheduling_{intra_layer_pipelining_type}_pipe_and_{intra_layer_parallelism_type}_parall.svg")

        plt.show()
    
    @staticmethod
    def plot_memory_usage(memory_traces, intra_layer_pipelining=False, intra_layer_parallelism=False, flash_attention=False, model_name="default"):
        if not memory_traces:
            print("No memory trace to plot.")
            return

        # Flatten if needed
        all_events = []
        for trace in memory_traces:
            if isinstance(trace, (tuple, list)) and len(trace) == 2:
                all_events.append(trace)
            elif isinstance(trace, list):
                for item in trace:
                    if isinstance(item, (tuple, list)) and len(item) == 2:
                        all_events.append(item)

        if not all_events:
            print("No memory trace to plot.")
            return

        # Sort by time to make sure the plot is clean
        all_events.sort(key=lambda x: x[0])
        times, usages = zip(*all_events)

        intra_layer_pipelining_type = "intra_layer" if intra_layer_pipelining else "None"
        intra_layer_parallelism_type = "intra_layer" if intra_layer_parallelism else "None"

        plt.figure(figsize=(12, 6))
        plt.plot(times, usages, label="TCDM Usage", color="tab:blue", marker='o', linewidth=2)

        # Add red horizontal line for max capacity
        tcdm_capacity = min(
            [globals()[n]["tcdm_capacity"] for n in ("PMCA_TILE_CONFIG","PMCA_RED_TILE_CONFIG","DA0_TILE_CONFIG")
            if n in globals() and globals()[n].get("num_tiles",0) > 0] or [0]
        )
        plt.axhline(y=tcdm_capacity, color='red', linestyle='--', linewidth=2, label=f"TCDM Capacity ({tcdm_capacity} KB)")

        plt.xlabel('Time (ms)', fontsize=12)
        plt.ylabel('TCDM Usage (KB)', fontsize=12)
        plt.title('TCDM Memory Usage Over Time (Attention)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        if flash_attention:
            plt.savefig(f"Plots/Attention/{model_name}_att_memory_usage_{intra_layer_pipelining_type}_pipe_{intra_layer_parallelism_type}_parall_FA.pdf")
        else:
            plt.savefig(f"Plots/Attention/{model_name}_att_memory_usage_{intra_layer_pipelining_type}_pipe_{intra_layer_parallelism_type}_parall.pdf")

        plt.show()



from collections import deque

class PrioritizedRollbackQueue:
    def __init__(self, max_depth=3):
        self.queue = deque()
        self.max_depth = max_depth

    def push(self, op):
        self.queue.append(op)
        if len(self.queue) > self.max_depth:
            self.queue.popleft()

    def pop_high_priority_ops(self):
        """Pop up to `max_depth` ops with highest rollback priority."""
        if not self.queue:
            return []

        # Rank by: memory allocated, not yet freed, and recency
        def rollback_priority(op):
            mem_alloc = sum(alloc["size"] for alloc in op.tcdm_alloc)
            mem_freed = len(op.tcdm_free)
            base_score = mem_alloc * (1 if mem_freed == 0 else 0.5)
            return base_score

        prioritized = sorted(list(self.queue), key=rollback_priority, reverse=True)
        rollback_set = prioritized[:self.max_depth]
        self.queue = deque([op for op in self.queue if op not in rollback_set])
        return rollback_set
