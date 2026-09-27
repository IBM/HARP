#!/usr/bin/env python3
#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import logging
import math
import os, sys
import time
from models.model_trial import *
from models.model_plot_utils import *
from scheduler.analog_scheduler import *
from utils.useful_funct import *
from utils.config import parse_args
from scheduler.scheduler import Scheduler
from nodes.cost_models import AttentionLatencyEstimator, AttentionLatencyEstimatorDAs
import warnings

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
# Import your CLI parser
def _wants_interactive_plots(argv):
    """True if --display_plots was passed a true-ish value (see utils.config.str2bool)."""
    for i, arg in enumerate(argv):
        if arg == '--display_plots' and i + 1 < len(argv):
            return argv[i + 1].lower() in ('true', '1', 't')
        if arg.startswith('--display_plots='):
            return arg.split('=', 1)[1].lower() in ('true', '1', 't')
    return False

# TkAgg needs a display, so only select it when plots are actually requested.
# Otherwise use the headless-safe Agg backend, so runs work over SSH/CI/Docker.
matplotlib.use("TkAgg" if _wants_interactive_plots(sys.argv[1:]) else "Agg")

colorama_init()

if any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    # We need cli_args in the same folder
    parse_args()
    sys.exit(0)

def main():
    args = parse_args()

    # --------- Print selected configuration ---------
    print("Selected configuration:")
    for name, value in vars(args).items():
        print(f"{name.upper()}: {value}")
    print()

    # ------------------ Configuration ------------------
    MODEL             = args.model                      #options: "MobileBERT, BERT_B"
    SL_ARG            = args.sl                         #Sequence Lenght: int
    AUTOREGRESSIVE    = args.autoregressive             #bool ["True", "False"]
    PREFILL_SIZE      = args.prefill_size               #int, only used for autoregressive models
    OPT_LEVEL         = args.opt_level                  #options: "latency", "area", "power", "area_latency_balance" 
    PARALL_TYPE       = args.parall_type                # Options: "none", "intra_layer"
    PIPE_TYPE         = args.pipe_type                  # Options: "none", "intra_layer", "inter_layer", "inter_intra_layer", "inter_block"
    FLASH_ATTENTION   = args.flash_attention            #True,  False
    SL_WARP           = args.sl_warp                    #int(128/2), None
    AUTO_TUNE_SL_WARP = args.auto_tune_sl_warp          #True, False
    CHUNK_OPT         = args.chunk_opt                  #should only be set to True to activate optimization when PIPE is inter_layer/intra_inter_layer/inter_block else, for intra_layer, is authomatically set to True
    CHUNK_SIZE_D      = args.chunk_size_d               #int, default 32
    USE_TRANSFER_LINK = args.use_transfer_link          #True, False
    DISPLAY_PLOTS     = args.display_plots              #True / False: set to False to disable plots and capture correct runtime
    DIGITAL_OPT       = args.digital_opt                #options: "latency", "power"
    DUMPLAYER_SIZE    = args.dumplayer_size             #int, default 128


    NUM_TILES = ACIM_TILE_CONFIG["num_tiles"]
    NUM_TIERS = ACIM_TILE_CONFIG["num_tiers"]

    # ------------------ Configuration checks ------------------
    if CHUNK_OPT == True and PIPE_TYPE not in ('inter_layer', 'inter_intra_layer', 'inter_block'):
        raise RuntimeError("Chunk optimization can only be used when using inter-layer pipelining: 'inter_layer', 'inter_intra_layer', 'inter_block'!")
    if AUTO_TUNE_SL_WARP == True and FLASH_ATTENTION == False:
        raise RuntimeError("SL warp autotune can only be used when using Flash Attention!")
    if PIPE_TYPE not in ("none", "intra_layer", "inter_layer", "inter_intra_layer", "inter_block"):
        raise RuntimeError("Select a supported Pipeline optimization among: 'none', 'intra_layer', 'inter_layer', 'inter_intra_layer', 'inter_block' ")
    if PARALL_TYPE not in ("none", "intra_layer"):
        raise RuntimeError("Select a supported Parallelism optimization among: 'none', 'intra_layer'")
    if OPT_LEVEL not in ("latency", "area", "power", "area_latency_balance"):
        raise RuntimeError("Select a supported analog optimization level among: 'latency', 'area', 'power', area_latency_balance'")
    SL_CROSS = SL_ARG
    # ------------------ Model Loading ------------------
    if MODEL == "MobileBERT":
        model_i, SL, num_blocks = get_model_mobileBERT(num_blocks=24, SL=SL_ARG)
        AUTOREGRESSIVE = False
    elif MODEL == "BERT_B":
        model_i, SL, num_blocks = get_model_BERT_B(num_blocks=12, SL=SL_ARG)
        AUTOREGRESSIVE = False
    elif MODEL == "ALBERT_B":
        model_i, SL, num_blocks = get_model_ALBERT_B(num_blocks=12, SL=SL_ARG)
        AUTOREGRESSIVE = False
    elif MODEL == "BERT_L":
        model_i, SL, num_blocks = get_model_BERT_L(num_blocks=24, SL=SL_ARG)
        AUTOREGRESSIVE = False
    elif MODEL == "dumpLayer":
        model_i, SL, num_blocks = get_model_dumpLayer(num_blocks=1, SL=SL_ARG, nrow=DUMPLAYER_SIZE, ncol=DUMPLAYER_SIZE)
        AUTOREGRESSIVE = False
    elif MODEL == "dumpModel":
        model_i, SL, num_blocks = get_model_dumpModel(num_blocks=2, SL=SL_ARG)
        AUTOREGRESSIVE = False
    elif MODEL == "NanoGPT":
        model_i, SL, num_blocks = get_model_NanoGPT(num_blocks=12, SL=SL_ARG)
        PREFILL_SIZE = PREFILL_SIZE + SL
    elif MODEL == "GPT2_small":
        model_i, SL, num_blocks = get_model_GPT2_small(num_blocks=12, SL=SL_ARG)
        PREFILL_SIZE = PREFILL_SIZE + SL
    elif MODEL == "T5_encoder":
        model_i, SL, num_blocks = get_model_T5_small_encoder(num_blocks=6, SL=SL_ARG)
        AUTOREGRESSIVE = False
    elif MODEL ==  "T5_decoder":
        model_i, SL, num_blocks, SL_CROSS = get_model_T5_small_decoder(num_blocks=6, SL=SL_ARG, SL_cross=512)
        PREFILL_SIZE = PREFILL_SIZE + SL
    else :
        raise ValueError("Unsupported model.")
    
    if MODEL in ["NanoGPT","GPT2_small", "T5_decoder"] and AUTOREGRESSIVE==True and (CHUNK_OPT == True or CHUNK_SIZE_D != 1):
        warnings.warn("Chunk optimization and chunk size different than 1 are not supported for autoregressive models! Setting CHUNK_OPT=False and CHUNK_SIZE_D=1.")
        CHUNK_OPT = False 
        CHUNK_SIZE_D = 1
    if CHUNK_SIZE_D > SL:
        warnings.warn(f"Chunk size D ({CHUNK_SIZE_D}) cannot be larger than sequence length SL ({SL})!\n Setting CHUNK_SIZE_D = SL = {SL}.")
        CHUNK_SIZE_D = SL
    if PREFILL_SIZE and PREFILL_SIZE<1 and AUTOREGRESSIVE:
        raise ValueError(f"Prefill size ({PREFILL_SIZE}) must be at least 1 for autoregressive models!")

    G = model_i.get_graph()
    if MODEL in ["T5_decoder"] and AUTOREGRESSIVE==True and PREFILL_SIZE > 1:
        if "emb_enc_out" in G:
            G.remove_node("emb_enc_out")
    share_map = detect_share_inputs(G)

    clear_logs(MODEL)

    # ------------------ Logger Setup ------------------
    log_suffix = f"{MODEL}_{SL}_{OPT_LEVEL}_opt"
    log_suffix += f"{'_intra_layer_parall' if PARALL_TYPE in ('intra_layer') else ''}{'_intra_layer_pipe' if PIPE_TYPE in ('intra_layer', 'inter_intra_layer') else ''}{'_inter_block' if PIPE_TYPE in ('inter_block') else ''}"
    log_suffix += f"{'_chunk' if PARALL_TYPE in ('intra_layer', 'both') else '_nochunk'}"
    log_suffix += f"{'_FA' if FLASH_ATTENTION else ''}{'_' + str(SL_WARP) if SL_WARP is not None and FLASH_ATTENTION==True else ''}"
    log_suffix += f"{'_autotune' if AUTO_TUNE_SL_WARP else ''}"
    log_suffix += f"{'_inter_layer_pipe' if PIPE_TYPE in ('inter_layer', 'inter_intra_layer') else ''}"
    log_suffix += f"{'_opt' if CHUNK_OPT else ''}"

    os.makedirs('outputs/Summary', exist_ok=True)
    os.makedirs('outputs/Scheduling', exist_ok=True)
    os.makedirs('outputs/Attention', exist_ok=True)
    os.makedirs('outputs/AnalogLayerMapping', exist_ok=True)
    os.makedirs('Plots/AnalogMapping', exist_ok=True)
    os.makedirs('Plots/Attention', exist_ok=True)
    os.makedirs('Plots/Scheduling', exist_ok=True)
    os.makedirs('Plots/Breakdown', exist_ok=True)
    os.makedirs('Plots/Model_graphs', exist_ok=True)
    
    draw_model_graph_html(
        model_i,
        out_name=f"{MODEL}"
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        encoding='utf-8',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'outputs/Summary/run_summary_{log_suffix}.txt', mode='w')
        ]
    )
    logger = logging.getLogger()

    #Starting execution
    start_time = time.time()

    #Save attention information
    if MODEL not in ("dumpLayer", "dumpModel", "T5_decoder") : #and MODEL != "T5_encoder":
        sl_full_tot, head_dim = G.nodes["atten_b0"]['layer'].input_shape
        _,hidden_size_tot = G.nodes["atten_b0"]['layer'].output_shape
    elif MODEL == "T5_decoder" : #or MODEL == "T5_encoder":
        sl_full_tot, head_dim = G.nodes["atten0_b0"]['layer'].input_shape
        _,hidden_size_tot = G.nodes["atten0_b0"]['layer'].output_shape
    else:
        sl_full_tot = 1
        hidden_size_tot = 0
        
    # ------------------ Scheduler setup------------------
    scheduler = Scheduler(
        optimization_level=OPT_LEVEL, 
        parallelism_type=PARALL_TYPE, 
        pipelining_type=PIPE_TYPE, 
        num_tiles=NUM_TILES, 
        num_tiers=NUM_TIERS,
        share_map=share_map, 
        model_name=MODEL, 
        SL=SL,
        flash_attention=FLASH_ATTENTION,
        sl_warp=SL_WARP,
        auto_tune_sl_warp = AUTO_TUNE_SL_WARP,
        autoregressive=AUTOREGRESSIVE, 
        prefill_size = PREFILL_SIZE,
        SL_CROSS = SL_CROSS,
        logger=logger,
        digital_opt=DIGITAL_OPT)

    # ----------------- Run Mapping & Scheduling -----------------
    ''' 1) Analog mapping '''
    mapping_summary, max_tile_used_single_tier, max_tile_used = scheduler.analog_mapping(G, DISPLAY_PLOTS)

    ''' 2) Create accelerator instances '''
    accelerator_instances = scheduler.create_accelerator_instances(mapping_summary=mapping_summary)

    ''' 3) Perform digital mapping '''
    scheduler.optimize_mapping(G, accelerator_instances)

    ''' 4) Schedule full hybrid system '''
    schedule_dict, total_lat, best_chunk_size, best_acc_inst = schedule_model_hybrid(
        G, 
        accelerator_instances, 
        mapping_summary, 
        SL=SL, 
        pipelining=scheduler.pipelining_type, 
        parallelism=scheduler.parallelism_type, 
        model_name=MODEL,
        chunk_optimization=CHUNK_OPT, 
        chunk_size_d=CHUNK_SIZE_D,
        analog_pool=scheduler.analog_pool,
        use_transfer_link=USE_TRANSFER_LINK,
        display_plots=DISPLAY_PLOTS,
        digital_opt=scheduler.digital_opt,
        SL_CROSS=SL_CROSS,
        prefill = PREFILL_SIZE
        )

    ''' 5) Save schedule summary '''
    summary_path = f'outputs/Scheduling/schedule_detailed_{log_suffix}.log'
    print(f"BEST chunk: {best_chunk_size}")
    save_schedule_txt(schedule_dict, summary_path)
    
    ''' 6) OPTIONAL: Plot attention scheduling details '''
    DA_KEY_RE = re.compile(r'/DA\d+\[')           # matches ".../DA0[", ".../DA1[", etc.
    PMCA_KEY_RE = re.compile(r'/PMCA(\b|\[)') 
    hidden_size = None
    sl_full = None
    attention_plot_done = False
    for node in nx.topological_sort(G):
        layer = G.nodes[node]['layer']

        if layer.type == "atten" and not attention_plot_done:
            _, hidden_size = layer.output_shape
            sl_full = getattr(layer, "parent_layer", layer).input_shape[0]
            def plot_leaf_attention_layers(attn_layer):
                if hasattr(attn_layer, "intra_layer_splits"):
                    for chunk in attn_layer.intra_layer_splits:
                        plot_leaf_attention_layers(chunk)
                        break
                else:
                    sl, head_d = attn_layer.input_shape
                    _, hidden_size = attn_layer.output_shape #this is the size of split layer for parallelism, so it is missing some info TODO
                    sl_full = getattr(attn_layer, "parent_layer", attn_layer).input_shape[0]

                    attn_name = getattr(attn_layer, "name", "atten")  # e.g., "atten_b0"
                    if attention_runs_on_DA(schedule_dict,DA_KEY_RE,PMCA_KEY_RE, attn_name):
                        attn_estimator = AttentionLatencyEstimatorDAs(
                            sl, head_d, hidden_size, MODEL,
                            max_size_vect_parall=DA0_TILE_CONFIG["max_parallel"]
                        )
                    else:
                        attn_estimator = AttentionLatencyEstimator(
                            sl, head_d, hidden_size, MODEL,
                            gemm_fast=True,
                            prefill_size=PREFILL_SIZE,
                            autoregressive=AUTOREGRESSIVE
                        )
                    #attn_estimator = AttentionLatencyEstimator(sl, head_d, hidden_size, MODEL, gemm_fast=True)
                    attn_estimator.sl_full = sl_full
                    attn_estimator.optimize_attention_chunk = scheduler.optimize_attention_chunk
                    attn_estimator.total_latency(intra_layer_pipelining=scheduler.intra_layer_pipelining, 
                                                intra_layer_parallelism=scheduler.intra_layer_parallelism, 
                                                inter_layer_pipelining=scheduler.inter_layer_pipelining,
                                                flash_attention=FLASH_ATTENTION, 
                                                sl_warp=SL_WARP, 
                                                auto_tune_sl_warp=AUTO_TUNE_SL_WARP,
                                                inter_layer_chunk_size=best_chunk_size,
                                                plot=DISPLAY_PLOTS, 
                                                plot_memory=DISPLAY_PLOTS)

            # Entry point: either a whole layer or set of sublayers
            if hasattr(layer, "parallel_sublayers"):
                for sub in layer.parallel_sublayers:
                    plot_leaf_attention_layers(sub)
                    break
            else:
                plot_leaf_attention_layers(layer)

            attention_plot_done = True

    # ----------------- Output & Summary -----------------
    print(f"{Fore.GREEN}"+f"{Style.BRIGHT} === Total latency: {total_lat:.6f} ms === {Style.RESET_ALL}")
    logging.info(f" === Total latency: {total_lat:.6f} ms === \n")
    if total_lat == float('inf'):
        print(f"{Fore.YELLOW}"+f"{Style.BRIGHT} === WARNING === {Style.RESET_ALL}")
        logging.info("Infinite latency detected! Check if the model fits into the DAs memory. Or set flash_attention flag to 'True'\n")
        exit(0)
    schedule_summary_and_plots(
        schedule_dict,
        best_acc_inst,
        scheduler=scheduler,
        model_name=MODEL,
        SL=SL,
        prefill=PREFILL_SIZE,
        total_latency=total_lat,
        mapping_summary=mapping_summary,
        num_blocks=num_blocks,
        sl_full=sl_full_tot,
        hidden_size=hidden_size_tot,
        max_tile_used_single_tier=max_tile_used_single_tier,
        max_tile_used=max_tile_used,
        display_plots = DISPLAY_PLOTS,
        autoregressive=AUTOREGRESSIVE,
        SL_CROSS = SL_CROSS
    )

    #Ending execution
    end_time = time.time()

    logging.info(f"\n=== Execution time: {end_time - start_time:.2f} seconds ===")
    print((Style.BRIGHT + Fore.WHITE+f"\n=== Execution time: {end_time - start_time:.2f} seconds ===\n")
)

if __name__ == '__main__':
        main()