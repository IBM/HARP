#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#

clock_frequency = 2*10**(-6) #2ns --> 0.000002 ms

# How many identical chips the platform can grow to.
NUM_CHIPS = 1 

# Configuration for ACIM and PMCA Tiles using ST 28nm
ACIM_TILE_CONFIG = {
    "tile_rows": 512,                   # Max rows per MVM
    "tile_cols": 512,                   # Max cols per MVM
    "num_tiers": 8,                     # Max cols per MVM    #mobileBERT: 8, BERT_B/BERT_L/NanoGPT: 8
    "integration_time": 350*10**(-6),   # Latency per tile in ms #Chip_N:350; Chip_P:40 
    "area": 1.5,                        # area in mm2
    "power": 0.07,                      #power in W #0.13W old; 0.07 ST-ACIM
    "power_ppu": 0.07*50/100/8,           #ppu power in W : 50% additional power for PPU
    "ppu_lat_initial": 3,               #clock cycles #Chip_N:3; Chip_P:1
    "n_ppus": 8,                        # number of ppu #Chip_N:8; Chip_P:64
    "clock": clock_frequency,           #ms #Chip_N:clock_frequency; Chip_P:clock_frequency/2
    "num_tiles": 96,                    #number of tile in the architecture   #mobileBERT: 16, BERT_B/NanoGPT: 96, BERT_L: 144
    "precision": 8                      #input/output precision (DAC/PPU out)
}

PMCA_TILE_CONFIG = { 
    "num_cores": 8,             # Max rows per MVM
    "redmule": "no",            # Max cols per MVM
    "area": 1.7,                # area in mm2
    "power": 0.192,             #power in W
    "clock": clock_frequency,   #ms
    "num_tiles": 20,             #number of tile in the architecture    #mobileBERT: 2, BERT_B/_L/NanoGPT: 20
    "dma_bandwidth": 8,         #Byte per cycle (B/cycle)
    "fp_precision": 16,         #FP16 precision
    "tcdm_capacity": 128        #TCDM size in KByte
}

PMCA_RED_TILE_CONFIG = {
    "num_cores": 8,             # Max rows per MVM
    "redmule": "yes",           # Max cols per MVM
    "area": 1.92,               # area in mm2
    "power": 0.258,             #power in W
    "clock": clock_frequency,   #ms
    "num_tiles": 12,             #number of tile in the architecture     #mobileBERT: 4, BERT_B/_L/NanoGPT: 12
    "dma_bandwidth": 8,         #Byte per cycle (B/cycle)
    "fp_precision": 16,         #FP16 precision
    "tcdm_capacity": 128        #TCDM size in KByte
}

DA0_TILE_CONFIG = {             # SFU 
    "num_cores": 4,             # Number of worker
    "const_delay": 0,           # Const Delay in cycles
    "max_parallel": 64,         # Max size of parallel operation per worker
    "area": 0.8047,             # area in mm2
    "power": 0.16,              #power in W: 0.160 + 5.5e-6 idle power
    "clock": clock_frequency, #ms
    "num_tiles": 0,             #number of tile in the architecture     #mobileBERT: 4, BERT_B/_L: 35 ; 12 
    "dma_bandwidth": 128*10/8,  #Byte per cycle (B/cycle)
    "fp_precision": 10,         #FP16 precision
    "tcdm_capacity": 1000       #TCDM size in KByte
}

SRAM_TILE_CONFIG = {
    "bandwidth": 64,       # Size in MiB/s
    "size": 1,              # Size in MiB # 0.025 and 0.05 work
    "area": 1.25,           # area in mm2
    "power": 0.00545,           #power in W
    "num_tiles": 8          #number of tile in the architecture
}

DDR_TILE_CONFIG = {
    "bandwidth": 800,       # Size in MiB/s  --> 800 MB/s, 1.6 GB/s, 3.2 GB/s, 6.4 GB/s
    "area": 1,              # area in mm2
    "power": 2.5,           #power in W
    "num_tiles": 1          #number of tile in the architecture
}

LINK_CONFIG = {
    "bandwidth_b_per_cycle": 512,  # bits per cycle
    "clock": clock_frequency,      # ms per cycle, same as other tiles
    "num_links": 1,                # you have exactly one physical link
    "area": 0,                     # area footprint 
    "power": 0.013                 # static power (W) 0.002
}