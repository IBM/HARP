#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from utils.chip_utils import dechip
import re


def plot_schedule_graph_patches(G, schedule_dict):
    """
    Draw the scheduled DAG using explicit Circle patches so that
    node colors are always honored.
    """
    # Build a color map for accelerator instances
    acc_names = list(schedule_dict.keys())
    cmap = cm.get_cmap('tab20', len(acc_names))
    acc_to_color = {acc: mcolors.to_hex(cmap(i)) for i, acc in enumerate(acc_names)}

    # Map layer-name -> graph node
    name_to_node = {data['layer'].name: node for node, data in G.nodes(data=True)}

    # Compute positions, sizes, and colors for each node
    pos = {}
    node_size_map = {}
    node_color_map = {}
    for acc in acc_names:
        lane = acc_names.index(acc)
        for layer_name, start, dur in schedule_dict[acc]:
            node = name_to_node[layer_name]
            x = start + dur / 2
            y = lane
            pos[node] = (x, y)
            node_size_map[node] = dur * 100  # scale factor
            node_color_map[node] = acc_to_color[acc]

    # Draw
    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw edges first
    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        arrowstyle='-|>',
        arrowsize=8,
        edge_color='gray',
        alpha=0.5
    )

    # Draw nodes with Circle patches
    for node, (x, y) in pos.items():
        size = node_size_map[node]
        color = node_color_map[node]
        radius = (size ** 0.5) * 0.1  # adjust scale for radius
        circ = plt.Circle((x, y), radius,
                          facecolor=color, edgecolor='black', linewidth=1, zorder=5)
        ax.add_patch(circ)

    # Draw labels on top
    for node, (x, y) in pos.items():
        ax.text(x, y, G.nodes[node]['layer'].name,
                ha='center', va='center', fontsize=8, zorder=6)

    # Axes settings
    ax.set_yticks(range(len(acc_names)))
    ax.set_yticklabels(acc_names)
    ax.set_xlabel("Time")
    ax.set_ylabel("Accelerator Instance")
    ax.set_title("Scheduled DAG: Node Color = Mapped Accelerator")
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # Legend for accelerators
    legend_patches = [plt.matplotlib.patches.Patch(color=acc_to_color[acc], label=acc)
                      for acc in acc_names]
    ax.legend(handles=legend_patches,
              title="Accelerator",
              loc='upper right',
              bbox_to_anchor=(1.15, 1.0))

    plt.tight_layout()
    plt.show()

def plot_schedule_timeline(schedule_dict,
                            model_name="default_model",
                            SL=1,
                            opt=None,
                            pipeline=None,
                            parallelism=None,
                            flash_attention=None,
                            collapse_analog=False):

    # 1) Optionally collapse all ACIMTile[...] into a single timeline
    if collapse_analog:
        analog_tasks = []
        for acc, tasks in schedule_dict.items():
            if acc.startswith("ACIMTile"):
                analog_tasks.extend(tasks)
        analog_tasks.sort(key=lambda x: x[1])
        seq = []
        t = 0.0
        for layer, _, dur in analog_tasks:
            seq.append((layer, t, dur))
            t += dur
        new_sched = {acc: tasks
                     for acc, tasks in schedule_dict.items()
                     if not acc.startswith("ACIMTile")}
        new_sched["Analog"] = seq
        schedule_dict = new_sched

    # 2) Plotting
    fig, ax = plt.subplots(figsize=(14, 6))

    # Gather unique layer-names by category
    analog_layers = sorted({layer for acc, tasks in schedule_dict.items()
                            if "Analog" in acc or "ACIMTile" in acc
                            for (layer, _, _) in tasks
                            if not layer.startswith("residual")})
    residual_layers = sorted({layer for acc, tasks in schedule_dict.items()
                              for (layer, _, _) in tasks
                              if layer.startswith("residual")})
    digital_layers  = sorted({layer for acc, tasks in schedule_dict.items()
                              if "PMCA" in acc
                              for (layer, _, _) in tasks})
    DDR_layers      = sorted({layer for acc, tasks in schedule_dict.items()
                              if "DDR" in acc
                              for (layer, _, _) in tasks})

    # Build colour maps
    analog_cmap   = cm.get_cmap('Blues',   len(analog_layers))
    residual_cmap = cm.get_cmap('Purples', len(residual_layers))
    digital_cmap  = cm.get_cmap('Oranges', len(digital_layers))
    DDR_cmap      = cm.get_cmap('Greens',  len(DDR_layers))

    layer_to_color = {}
    for i, l in enumerate(analog_layers):   layer_to_color[l] = analog_cmap(i)
    for i, l in enumerate(residual_layers): layer_to_color[l] = residual_cmap(i)
    for i, l in enumerate(digital_layers):  layer_to_color[l] = digital_cmap(i)
    for i, l in enumerate(DDR_layers):      layer_to_color[l] = DDR_cmap(i)

    # plot bars
    y_labels_tmp = list(schedule_dict.keys())
    y_labels = [s for s in y_labels_tmp if not dechip(s).startswith("SRAM")]

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    for idx, acc in enumerate(y_labels):
        for layer, start, dur in schedule_dict[acc]:
            ax.broken_barh([(start, dur)], (idx-0.4, 0.8),
                           facecolors=layer_to_color.get(layer,'gold'),
                           edgecolors='black')
            ax.text(start + dur/2, idx, layer,
                    ha='center', va='center', fontsize=8)

    ax.set_xlabel("Time (ms)")
    ax.set_title("Scheduled Layers on Accelerator Instances")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    # create patches
    analog_patches   = [mpatches.Patch(color=layer_to_color[l], label=l) for l in analog_layers]
    other_patches    = ([mpatches.Patch(color=layer_to_color[l], label=l) for l in digital_layers] +
                        [mpatches.Patch(color=layer_to_color[l], label=l) for l in DDR_layers] +
                        [mpatches.Patch(color=layer_to_color[l], label=l) for l in residual_layers])

    # top legend: analog only
    top_leg = ax.legend(handles=analog_patches,
                        title="Analog Layers",
                        loc='upper right',
                        bbox_to_anchor=(1.2, 1.01),
                        fontsize=9,
                        ncol=2)
    ax.add_artist(top_leg)

    # bottom legend: digital + DDR + residual
    # place it centered below the plot
    fig.legend(handles=other_patches,
               title="Digital / DDR / Residual",
               loc='lower center',
               ncol= max(1, len(other_patches)//2),
               bbox_to_anchor=(0.5, -0.005),
               fontsize=9)

    plt.tight_layout(rect=(0,0.15,0.85,1)) 
    # make room below for the bottom legend
    plt.savefig(f"Plots/Scheduling/scheduling_{model_name}_{SL}_{pipeline}_pipe_{parallelism}_parall_{flash_attention}_FA_{opt}_opt.pdf", bbox_inches='tight')
    plt.savefig(f"Plots/Scheduling/scheduling_{model_name}_{SL}_{pipeline}_pipe_{parallelism}_parall_{flash_attention}_FA_{opt}_opt.svg", bbox_inches='tight', dpi=300)
    plt.show()