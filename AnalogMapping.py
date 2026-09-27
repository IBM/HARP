#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#

import math
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from nodes.accelerator_config import ACIM_TILE_CONFIG
from collections import defaultdict
from matplotlib import colormaps

# --- Configuration --------------------------------------------------------------
TILE_ROWS  = ACIM_TILE_CONFIG["tile_rows"]
TILE_COLS  = ACIM_TILE_CONFIG["tile_cols"]
NUM_TILES  = ACIM_TILE_CONFIG["num_tiles"]
NUM_TIERS  = ACIM_TILE_CONFIG["num_tiers"]
TILE_POWER = ACIM_TILE_CONFIG["power"]
PPU_OVER   = ACIM_TILE_CONFIG["power_ppu"]
TILE_LAT   = ACIM_TILE_CONFIG["integration_time"]
PPU_LAT    = ACIM_TILE_CONFIG["ppu_lat_initial"]
CLOCK      = ACIM_TILE_CONFIG["clock"]
N_PPU      = ACIM_TILE_CONFIG["n_ppus"]


# --- Analog Tier & Tile Classes -------------------------------------------------

class AnalogTier:
    def __init__(self, tile_id, tier_id):
        self.tile_id     = tile_id
        self.tier_id     = tier_id
        self.usage_map   = [[False]*TILE_COLS for _ in range(TILE_ROWS)]
        self.used_blocks = []  # (layer, (r0,r1), (c0,c1))

    def has_space_for(self, r0, r1, c0, c1):
        for r in range(r0, r1):
            for c in range(c0, c1):
                if self.usage_map[r][c]:
                    return False
        return True

    def allocate(self, layer, r0, r1, c0, c1):
        for r in range(r0, r1):
            for c in range(c0, c1):
                self.usage_map[r][c] = True
        self.used_blocks.append((layer, (r0, r1), (c0, c1)))

    def free(self, layer, r0, r1, c0, c1):
        # clear bits
        for r in range(r0, r1):
            for c in range(c0, c1):
                self.usage_map[r][c] = False
        # remove only the block that matches both layer AND region
        self.used_blocks = [
            (lay, rr, cc)
            for (lay, rr, cc) in self.used_blocks
            if not (lay == layer and rr == (r0, r1) and cc == (c0, c1))
        ]



class TieredAnalogTile:
    def __init__(self, tile_id, num_tiers):
        self.tile_id = tile_id
        self.tiers   = [AnalogTier(tile_id, t) for t in range(num_tiers)]


# --- General Pool ----------------------------------------------------------------

class GeneralAnalogTilePool:
    def __init__(self,
                 num_tiles=NUM_TILES,
                 num_tiers=NUM_TIERS,
                 optimize_for='latency',
                 share_map=None,
                 chip_id : int = 0):
        """
        optimize_for:
          - 'area'               : reuse any free space
          - 'latency'            : reuse only among true share-input peers
          - 'area_latency_balance': reuse any tile, but for share-input peers
                                     enforce no column-overlap
        """
        assert optimize_for in ('area','latency','area_latency_balance')
        self.optimize_for      = optimize_for
        self.share_map         = share_map or {}
        self.num_tiers         = num_tiers
        self.tiles             = [
            TieredAnalogTile(i, num_tiers)
            for i in range(num_tiles)
        ]
        self.layer_map         = {}  # layer_name -> list of (tile,tier,r0,r1,c0,c1)
        self.layer_input_shape = {}  # layer_name -> (in_rows,out_cols)
        self.allocated_groups = set()    # NEW: track (prefix,blk) we've already group-placed
        self.chip_id = chip_id

    def split_layer(self, in_rows, out_cols):
        blocks = []
        for r in range(0, in_rows, TILE_ROWS):
            re = min(r + TILE_ROWS, in_rows)
            for c in range(0, out_cols, TILE_COLS):
                ce = min(c + TILE_COLS, out_cols)
                blocks.append((r, re, c, ce))
        blocks.sort(key=lambda x: (x[1]-x[0])*(x[3]-x[2]), reverse=True)
        return blocks

    def try_allocate_layer(self, layer_name, in_rows, out_cols):
        """
        1) Split into sub-blocks.
        2) If blk>0, do your group-first carve exactly as before.
        3) Otherwise (or if that fails) do two-stage packing:
        a) REUSE-PASS (area ignores peers entirely)
        b) FRESH-SLOT FALLBACK (uses each sub-block's own coords)
        """
        import re

        # helper to turn e.g. "fc2_b1" -> "fc2_b0"
        def base_block0(n):
            return re.sub(r'_b\d+$', '_b0', n)

        # 0) split & record
        blocks = self.split_layer(in_rows, out_cols)
        self.layer_input_shape[layer_name] = (in_rows, out_cols)

        # detect prefix+blk
        m = re.match(r"(.+)_b(\d+)$", layer_name)
        if m:
            prefix, blk = m.group(1), int(m.group(2))
        else:
            prefix, blk = layer_name, 0

        # A) group-first carve for block N>0
        if blk > 0:
            peers = {
                p for p in self.share_map.get(layer_name, {layer_name})
                if p.startswith(("fc", "emb_lin"))
            }
            group = sorted(Name for Name in peers if Name.endswith(f"_b{blk}"))
            if layer_name not in group:
                group.insert(0,layer_name)
            # 2) only carve if *none* of them is already placed
            if group and not any(g in self.layer_map for g in group):
                # 3) each member has identical sub-blocks layout
                group_blocks = self.split_layer(in_rows, out_cols)

                shapes = { 
                        g: self.layer_input_shape[base_block0(g)] 
                        for g in group 
                        }
                g_cols = sum(out_cols for _, out_cols in shapes.values())

                # 4) try each prior block's pattern (most recent first)
                for k in range(blk-1, -1, -1):
                    base = f"{prefix}_b{k}"
                    if base not in self.layer_map:
                        continue
                    prior_slots = self.layer_map[base]   # list of (t,tier,br,er,bc,ec)
                    if len(prior_slots) != len(group_blocks):
                        continue

                    cand = []
                    ok = True
                    # carve each sub-block into same tile/tier, rows->cols
                    for idx,(r0,r1,c0,c1) in enumerate(group_blocks):
                        h, w = r1 - r0, c1 - c0
                        t0, tier0, pr_br, pr_er, old_bc, old_ec = prior_slots[idx]
                        tier = self.tiles[t0].tiers[tier0]

                        placed = False
                        # a) slide along rows first
                        for br in range(0, TILE_ROWS - h + 1, 16):
                            if tier.has_space_for(br, br + h, old_bc, old_ec):
                                cand.append((t0, tier0, br, br + h, old_bc, old_ec))
                                placed = True
                                break

                        # b) if rows didn't fit, slide along columns but keep original rows (pr_br..pr_er)
                        if not placed:
                            for bc in range(0, TILE_COLS - g_cols + 1, 16):
                                if tier.has_space_for(pr_br, pr_er, bc, bc + g_cols):
                                    cand.append((t0, tier0, pr_br, pr_er, bc, bc + w))
                                    placed = True
                                    break

                        if not placed:
                            ok = False
                            break

                    if not ok:
                        cand.clear()
                        continue

                    # 5) commit all carve-ins at once #TODO CHECK before open-source
                    for g in group:
                        self.layer_map[g] = []
                        for slot in cand:
                            t0,t1,br,er,bc,ec = slot
                            if g == layer_name:
                                self.tiles[t0].tiers[t1].allocate(g, br, er, bc, ec)
                                self.layer_map[g].append(slot)
                    
                    # done--return the slots for *this* layer
                    return self.layer_map[layer_name]

                            
        
        # B) two-stage packing
        placements = []
        peers = {
                p for p in self.share_map.get(layer_name, set())
                if p.startswith(("fc", "emb_lin"))
            }
        for (r0, r1, c0, c1) in blocks:
            h, w   = (r1 - r0), (c1 - c0)
            placed = False

            # 1) REUSE-PASS
            for tier_id in range(self.num_tiers):
                if placed: break
                for tile in self.tiles:
                    tier = tile.tiers[tier_id]

                    # a) allowed?
                    if   self.optimize_for == 'area':
                        allow = bool(tier.used_blocks)
                    elif self.optimize_for == 'latency':
                        allow = any(lhs in peers for (lhs,_,_) in tier.used_blocks)
                    else:  # area_latency_balance
                        allow = bool(tier.used_blocks)

                    if not allow:
                        continue

                    # b) peer-cols only in area_latency_balance
                    peer_cols = []
                    if self.optimize_for == 'area_latency_balance':
                        for lhs,_,(ec0,ec1) in tier.used_blocks:
                            if lhs in peers:
                                peer_cols.append((ec0,ec1))

                    # c) choose row-starts
                    if self.optimize_for == 'area' or not peers:
                        # area ignores peers; also if no peers, full sweep
                        row_starts = range(0, TILE_ROWS - h + 1, 16)
                    else:
                        # latency or balance: only the exact peer-row offsets
                        row_starts = sorted({ r0_ for lhs,(r0_,r1_),_
                                            in tier.used_blocks if lhs in peers })

                    # d) scan for a fit
                    for br in row_starts:
                        if placed: break
                        for bc in range(0, TILE_COLS - w + 1, 16):
                            # enforce no col-overlap if balancing
                            if peer_cols:
                                nc0, nc1 = bc, bc + w
                                if any(not (nc1 <= p0 or p1 <= nc0)
                                    for (p0,p1) in peer_cols):
                                    continue

                            if tier.has_space_for(br, br+h, bc, bc+w):
                                tier.allocate(layer_name, br, br+h, bc, bc+w)
                                placements.append((tile.tile_id, tier_id,
                                                br, br+h, bc, bc+w))
                                placed = True
                                break
                        # end bc
                    if placed:
                        break
                # end tile
            # end tier

            # 2) FRESH-SLOT FALLBACK (use the sub-block's own coords)
            if not placed:
                for tier_id in range(self.num_tiers):
                    if placed: break
                    for tile in self.tiles:
                        tier = tile.tiers[tier_id]
                        if tier.has_space_for(0, h, 0, w):
                            tier.allocate(layer_name, 0, h, 0, w)
                            placements.append((tile.tile_id, tier_id,
                                            0, h, 0, w))
                            placed = True
                            break
                    # end tile
                # end tier

            if not placed:
                raise RuntimeError(f"Cannot place sub-block {(r0, r1, c0, c1)} "
                                f"of {layer_name}")

        # commit & return
        self.layer_map[layer_name] = placements
        return placements


    def estimate_latency(self, layer_name, SL=1, share_map=None, placements=None):
        in_r, in_c = self.layer_input_shape[layer_name]
        if layer_name.startswith("residual"):
            return self.estimate_latency_residual(layer_size=in_r, SL=1, ext_tile_acc=False)
        else:
            nrow    = math.ceil(in_r / TILE_ROWS)
            ncols   = in_c if in_c <= TILE_COLS else TILE_COLS
            
            if nrow == 1:
                base    = TILE_LAT + (PPU_LAT + ncols/N_PPU) * CLOCK 
                ppu_lat = (PPU_LAT + ncols/N_PPU) * CLOCK 
            else:
                #Optimized for multiple rows TODO: update HARP results.
                base    = TILE_LAT + ((PPU_LAT + ncols/N_PPU) * CLOCK) + (PPU_LAT+1) * (nrow-1) * CLOCK
                ppu_lat = ((PPU_LAT + ncols/N_PPU) * CLOCK) + (PPU_LAT+1) * (nrow-1) * CLOCK
            

        extra_base = 0.0
        extra_ppu = 0.0

        if placements:
            groups = {}
            for p in placements:
                # expect (tile_id, tier_id, row_in, row_end, col_in, col_end)
                tid, tier_id, r_in, r_end, c_in, c_end = p
                groups.setdefault((tid, tier_id), []).append((r_in, r_end, c_in, c_end))

            for key, slots in groups.items():
                # first slot is already accounted for in base/ppu_lat;
                # add latency for the remaining slots
                for (r_in, r_end, c_in, c_end) in slots[1:]:
                    col_width = (c_end - c_in)
                    add_base = TILE_LAT + (PPU_LAT + col_width / N_PPU) * CLOCK
                    add_ppu  = (PPU_LAT + col_width / N_PPU) * CLOCK
                    extra_base += add_base
                    extra_ppu += add_ppu
                    base += add_base
                    ppu_lat += add_ppu

        return (
            round(base * SL, 9),
            round(ppu_lat * SL, 9),
            round(extra_base * SL, 9),
            round(extra_ppu * SL, 9)
        ) # base = ACIM latency + PPU latency; ppu_lat = PPU latency only
    
    def estimate_latency_residual(self, layer_size, SL=1, ext_tile_acc=False, not_from_analog=False):
        if ext_tile_acc: # and layer_size <= TILE_ROWS:
            return PPU_LAT * SL * CLOCK #ms
        elif not_from_analog:
            return (PPU_LAT + layer_size/N_PPU) * SL *CLOCK #ms
        else:
            return (PPU_LAT + TILE_COLS/N_PPU) * SL *CLOCK #ms


    def plot(self, rows=None, cols=None, model_name="default_model"):
        """
        Like plot_better_layout, but each (tile,tier) gets its own cell.
        """
        # total number of subplot cells = num_tiles * num_tiers
        total_slots = len(self.tiles) * self.num_tiers

        # infer rows/cols if missing
        if rows is None and cols is None:
            rows = 2
            cols = math.ceil(total_slots / rows)
        elif rows is not None and cols is None:
            cols = math.ceil(total_slots / rows)
        elif cols is not None and rows is None:
            rows = math.ceil(total_slots / cols)

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 3.5, rows * 5),
            sharey=True,
            squeeze=False
        )
        fig.suptitle(f"Analog MVM allocation -- Chip {self.chip_id}", fontsize=12)

        # Use only colormaps that are perceptually diverse and work well with categorical data
        available_maps = [
            'Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'Greys',
            'YlGn', 'YlOrBr', 'PuBuGn', 'BuPu', 'GnBu', 'YlGnBu', 'PuRd',
            'RdPu', 'OrRd', 'cividis', 'viridis', 'plasma', 'magma', 'cubehelix'
        ]

        # fallback to tab20 or random colors if blocks > len(available_maps)
        # Step 2: map each block to a palette and assign unique shades per layer
        block_to_layers = defaultdict(list)
        for l in sorted({l for t in self.tiles for tier in t.tiers for l,_,_ in tier.used_blocks}):
            if "_b" in l:
                block_id = l.split("_b")[-1]
                block_to_layers[block_id].append(l)
            else:
                block_to_layers["misc"].append(l)  # fallback if no block tag

        layer_color = {}
        sorted_blocks = sorted(block_to_layers.items(), key=lambda x: x[0])
        num_maps = len(available_maps)

        for i, (block_id, layers) in enumerate(sorted_blocks):
            cmap_name = available_maps[i % num_maps]
            cmap = cm.get_cmap(cmap_name, len(layers))
            for j, layer in enumerate(sorted(layers)):
                layer_color[layer] = cmap(j)

        # now plot each (tile,tier) in one cell
        for tile in self.tiles:
            for tier in tile.tiers:
                # compute the 1D index of this slot
                slot_idx = tile.tile_id * self.num_tiers + tier.tier_id
                row = slot_idx // cols
                col = slot_idx % cols
                ax = axes[row][col]

                ax.set_aspect('equal')
                ax.set_xlim(0, TILE_COLS)
                ax.set_ylim(0, TILE_ROWS)
                ax.invert_yaxis()
                ax.set_title(f"Chip{self.chip_id} T{tile.tile_id},Tr{tier.tier_id}", fontsize=8)
                ax.set_xlabel("Cols", fontsize=6)
                if col == 0:
                    ax.set_ylabel("Rows", fontsize=6)

                # draw each block
                for l,(r0,r1),(c0,c1) in tier.used_blocks:
                    rect = patches.Rectangle(
                        (c0,r0),
                        c1-c0,
                        r1-r0,
                        linewidth=1,
                        edgecolor='black',
                        facecolor=layer_color[l]
                    )
                    ax.add_patch(rect)
                    ax.text(
                        c0 + (c1-c0)/2,
                        r0 + (r1-r0)/2,
                        l,
                        ha='center', va='center',
                        fontsize=6,
                        color='white' if layer_color[l][0]+layer_color[l][1]+layer_color[l][2]<1.5 else 'black'
                    )

        # turn off any unused cells
        for idx in range(total_slots, rows*cols):
            r = idx // cols
            c = idx % cols
            axes[r][c].axis('off')

        # legend
        patches_ = [patches.Patch(color=color, label=l) for l,color in layer_color.items()]
        # fig.legend(handles=patches_, loc='upper center', ncol=min(len(patches_), 5))
        fig.legend(handles=patches_, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=round(len(patches_)/math.sqrt(ACIM_TILE_CONFIG['num_tiles'])))

        plt.tight_layout(rect=[0,0,1,0.90])
        plt.savefig(f"Plots/AnalogMapping/{model_name}_analog_mapping_{self.optimize_for}_chip{self.chip_id}.pdf")
        plt.show()


    # --- Terminal Debug Print ---------------------------------------------------
    def debug_print_terminal(self):
        print("\n\033[1mLayer Mapping (tile, tier):\033[0m")
        for t in self.tiles:
            for tier in t.tiers:
                if not tier.used_blocks:
                    continue
                print(f"Tile {t.tile_id}, Tier {tier.tier_id}")
                print("-" * 30)
                for l,(r0,r1),(c0,c1) in tier.used_blocks:
                    print(f"  {l:10s} rows({r0},{r1}) cols({c0},{c1})")
                print("=" * 30)

    # --- File Debug Print --------------------------------------------------------
    def debug_print(self, model_name="default_model"):
        fn = f"outputs/AnalogLayerMapping/{model_name}_AnalogLayerMappingDebug_{self.optimize_for}.txt"
        with open(fn, "w") as f:
            f.write("Layer Mapping (tile, tier):\n")
            f.write("="*50 + "\n")
            for t in self.tiles:
                for tier in t.tiers:
                    if not tier.used_blocks:
                        continue
                    f.write(f"Tile {t.tile_id}, Tier {tier.tier_id}\n")
                    f.write("-"*30 + "\n")
                    for l,(r0,r1),(c0,c1) in tier.used_blocks:
                        f.write(f"  {l:10s} rows({r0},{r1}) cols({c0},{c1})\n")
                    f.write("="*30 + "\n")
        print(f"Debug saved to {fn}")
    
    def summarize_mapping_inferred(self,
                                   tile_power=TILE_POWER,
                                   model_name="default_model",
                                   SL=1):
        """
        Multi-tier summary in the same style as your one-tier version,
        plus counts of tiles used at tier 0 and across all tiers.
        """
        layer_summary = {}
        fn = f"outputs/AnalogLayerMapping/{model_name}_AnalogLayerMappingSummary_{self.optimize_for}_opt.txt"
        utilization_percent = {}

        with open(fn, "w") as f:
            f.write("\nLayer Mapping Summary:\n")
            f.write("=" * 50 + "\n\n")

            # 1) Per-layer data
            for layer_name in sorted(self.layer_map.keys()):
                # accumulate per-tile usage (summing all tiers)
                is_residual = layer_name.startswith("residual")
                layer_size = self.layer_map[layer_name]['input_shape'] if is_residual else None
                tile_usage   = {}   # tile_id -> used area
                total_power  = 0.0

                if not is_residual:
                    for tile in self.tiles:
                        tier_area = TILE_ROWS * TILE_COLS
                        used_area = 0
                        for tier in tile.tiers:
                            for l, (r0, r1), (c0, c1) in tier.used_blocks:
                                if l == layer_name:
                                    used_area += (r1 - r0) * (c1 - c0)

                        if used_area > 0:
                            tile_usage.setdefault(tile.tile_id, 0)
                            tile_usage[tile.tile_id] += used_area
                            total_power += (used_area / tier_area) * tile_power #Power of ACIM tiles involved in the computations for that layer (no PPU)

                    utilization_percent = {
                        tid: round(100 * used / (TILE_ROWS * TILE_COLS), 2)
                        for tid, used in tile_usage.items()
                    }
                    latency, ppu_latency, extra_base, extra_ppu = self.estimate_latency(
                        layer_name, SL, placements=self.layer_map[layer_name]
                    )

                    # build col_blocks: tile_id -> list of column-ranges (across tiers)
                    col_blocks = {}
                    for tile in self.tiles:
                        for tier in tile.tiers:
                            for l, (_, _), (c0, c1) in tier.used_blocks:
                                if l == layer_name:
                                    col_blocks.setdefault(tile.tile_id, []).append((c0, c1))
                    
                    # collect the slot list from self.layer_map
                    slots = [(tid, tier) for tid, tier, *_ in self.layer_map[layer_name]]

                    extra_estimated_power = 0.0
                    extra_ppu_group_count = 0.0
                    if self.layer_map[layer_name]:
                        tile_groups = {}
                        for p in self.layer_map[layer_name]:
                            tid, tier_id, r0, r1, c0, c1 = p
                            tile_groups.setdefault((tid, tier_id), []).append((r0, r1, c0, c1))

                        for group_slots in tile_groups.values():
                            for idx, (r0, r1, c0, c1) in enumerate(group_slots):
                                if idx == 0:
                                    continue
                                extra_estimated_power += ((r1 - r0) * (c1 - c0)) / tier_area * tile_power
                            if len(group_slots) > 1:
                                extra_ppu_group_count += 1
                    ##########
                    estimated_power_ppu = round(PPU_OVER * N_PPU * len(slots), 3)
                    extra_ppu_power = round(PPU_OVER * N_PPU * extra_ppu_group_count, 3)
                    layer_summary[layer_name] = {
                        "slots":                    sorted(slots),
                        "tiles_used":               sorted(tile_usage.keys()),
                        "tile_utilization_percent": utilization_percent,
                        "latency":                  latency,
                        "ppu_latency":              ppu_latency,
                        "extra_base":               extra_base,
                        "extra_ppu":                extra_ppu,
                        "estimated_power":          round(total_power, 3),
                        "extra_estimated_power":    round(extra_estimated_power, 3),
                        "estimated_power_ppu":      estimated_power_ppu,
                        "extra_ppu_power":          round(extra_ppu_power, 3),
                        "col_blocks":               col_blocks,
                    }
                else:
                    # Residual layers -- tile and latency were pre-computed during scheduling

                    self.layer_map[layer_name] = {}
                    power_ppu_tmp =  PPU_OVER*N_PPU*math.ceil(layer_size/TILE_COLS)

                    layer_summary[layer_name] = {
                        "slots": {},
                        "tiles_used": {},
                        "tile_utilization_percent": {},
                        "latency": None,
                        "ppu_latency": None,
                        "extra_base":               None,
                        "extra_ppu":                None,
                        "estimated_power": power_ppu_tmp, # no ACIM power, only PPU (actually not used for residuals)
                        "extra_estimated_power":   power_ppu_tmp, # no ACIM power, only PPU (actually not used for residuals)
                        "estimated_power_ppu": power_ppu_tmp,
                        "extra_ppu_power": 0.0,
                        "col_blocks": {},
                    } #TODO
                
                if utilization_percent:
                    # write
                    f.write(f"Layer: {layer_name}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  Tiles Used: {layer_summary[layer_name]['tiles_used']}\n")
                    f.write("  Tile Utilization (%):\n")
                    for tid, pct in utilization_percent.items():
                        f.write(f"    Tile {tid}: {pct}%\n")
                    f.write("  Column Blocks:\n")
                    for tid, ranges in col_blocks.items():
                        f.write(f"    Tile {tid}: {ranges}\n")
                    f.write(f"  Latency: {latency} ms\n")
                    f.write(f"  Extra ACIM Latency: {layer_summary[layer_name]['extra_base']} ms\n")
                    f.write(f"  Extra PPU Latency: {layer_summary[layer_name]['extra_ppu']} ms\n")
                    f.write(f"  PPU Latency: {ppu_latency} ms\n")
                    f.write(f"  Estimated Power: {layer_summary[layer_name]['estimated_power']} W\n")
                    f.write(f"  Extra Estimated Power: {layer_summary[layer_name]['extra_estimated_power']} W\n")
                    f.write(f"  Extra PPU Power: {layer_summary[layer_name]['extra_ppu_power']} W\n")
                    f.write("=" * 50 + "\n\n")

            # 2) Compute tile counts
            #    a) tier 0
            tier0_tiles = {
                tid
                for tile in self.tiles
                for tid in [tile.tile_id]
                for blk in tile.tiers[0].used_blocks
                if blk[0] in self.layer_map  # any layer
            }
            num_tier0_tiles = len(tier0_tiles)

            all_tiles = {
                (tid,tier_id)
                for layer, places in self.layer_map.items()
                if not layer.startswith("residual")
                for (tid, tier_id, *_) in places
            }

            num_all_tiles = len(all_tiles)

            f.write(f"Number of used Tiles in Tier 0: {num_tier0_tiles}\n")
            f.write(f"Total Tiles/Tier used:    {num_all_tiles}\n")

        print(f"Summary saved to {fn}")
        return layer_summary, num_tier0_tiles, num_all_tiles
