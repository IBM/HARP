#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#

import html
import networkx as nx


def layer_fillcolor(layer_type: str) -> str:
    colors = {
        "fc": "#abcef2",
        "embedding": "#9dd3a1",
        "atten": "#da5c5c",
        "residual": "#a383a3",
        "layernorm": "#ecaf96",
        "gelu": "#fffd9a",
        "softmax": "#89ebe6",
    }
    return colors.get(layer_type, "#c6c6c6")


def fmt_shape(shape):
    if shape is None:
        return "-"
    if isinstance(shape, tuple):
        return "x".join(str(x) for x in shape)
    return str(shape)


def compute_vertical_layout(G, row_gap=140, col_gap=260):
    """
    Top-to-bottom layout based on DAG levels.
    Parallel nodes go side-by-side.
    """
    order = list(nx.topological_sort(G))

    level = {}
    for node in order:
        preds = list(G.predecessors(node))
        if not preds:
            level[node] = 0
        else:
            level[node] = max(level[p] for p in preds) + 1

    nodes_by_level = {}
    for n, lv in level.items():
        nodes_by_level.setdefault(lv, []).append(n)

    pos = {}
    for lv in sorted(nodes_by_level):
        nodes = nodes_by_level[lv]
        n = len(nodes)

        for i, node in enumerate(nodes):
            x = (i - (n - 1) / 2) * col_gap
            y = lv * row_gap
            pos[node] = (x, y)

    return pos


def vertical_edge_path(x1, y1, x2, y2):
    """
    Smooth vertical connection
    """
    my = (y1 + y2) / 2
    return f"M {x1},{y1} C {x1},{my} {x2},{my} {x2},{y2}"


def draw_model_graph_html(model, out_name="model_graph"):
    G = model.get_graph()
    pos = compute_vertical_layout(G)

    node_w = 220
    node_h = 80
    margin = 100

    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = int(max_x - min_x + node_w + margin * 2)
    height = int(max_y - min_y + node_h + margin * 2)

    # shift positions to positive space
    shift_x = -min_x + margin
    shift_y = margin

    svg_parts = []

    # Arrow
    svg_parts.append("""
    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8 z" fill="#666"/>
      </marker>
    </defs>
    """)

    # Edges
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        start_x = x1 + shift_x
        start_y = y1 + node_h + shift_y

        end_x = x2 + shift_x
        end_y = y2 + shift_y

        path = vertical_edge_path(start_x, start_y, end_x, end_y)

        svg_parts.append(
            f'<path d="{path}" fill="none" stroke="#666" stroke-width="2" marker-end="url(#arrow)"/>'
        )

    # Nodes
    for node in G.nodes():
        x, y = pos[node]
        x += shift_x - node_w / 2
        y += shift_y

        layer = G.nodes[node].get("layer", None)

        if layer:
            name = layer.name
            ltype = layer.type
            in_shape = fmt_shape(layer.input_shape)
            out_shape = fmt_shape(layer.output_shape)
        else:
            name, ltype, in_shape, out_shape = node, "unknown", "-", "-"

        fill = layer_fillcolor(ltype)

        svg_parts.append(f"""
        <g>
          <rect x="{x}" y="{y}" width="{node_w}" height="{node_h}"
                rx="10" ry="10"
                fill="{fill}" stroke="#333" stroke-width="1.5"/>
          <text x="{x+12}" y="{y+22}" font-size="14" font-weight="700">{html.escape(name)}</text>
          <text x="{x+12}" y="{y+42}" font-size="11">type: {html.escape(ltype)}</text>
          <text x="{x+12}" y="{y+60}" font-size="11">
            in: {html.escape(in_shape)} -> out: {html.escape(out_shape)}
          </text>
        </g>
        """)

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
      {"".join(svg_parts)}
    </svg>
    """

    html_page = f"""<!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>{model.name}</title>
    <style>
    body {{
        margin: 0;
        font-family: Arial;
        background: #f5f6f8;
    }}
    .header {{
        padding: 14px;
        font-size: 20px;
        font-weight: bold;
        background: white;
        border-bottom: 1px solid #ddd;
    }}
    .viewport {{
        height: calc(100vh - 50px);
        overflow: auto;
        padding: 20px;
    }}
    .canvas {{
        background: white;
        padding: 20px;
        border: 1px solid #ddd;
    }}
    </style>
    </head>
    <body>
    <div class="header">{model.name} (vertical view)</div>
    <div class="viewport">
    <div class="canvas">
    {svg}
    </div>
    </div>
    </body>
    </html>
    """

    path = f"Plots/Model_graphs/{out_name}_graph.html"
    with open(path, "w") as f:
        f.write(html_page)

    print(f"Saved: {path}")
    
