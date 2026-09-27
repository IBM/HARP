#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#

import networkx as nx

# ============================================================
# Layer naming assumptions
# ============================================================

# - The first fully connected layer inside each block is always named `fc1`.
# - Fully connected layers that appear before the blocks do not include the
#   suffix `"_b"`.
#   Example: `emb_lin` in MobileBERT
# - Fully connected layers that appear inside or after the blocks may include
#   the suffix `"_bX"`, where X is the block index.
#   Example: `fc5_b0` in NanoGPT


# ============================================================
# Layer size conventions
# ============================================================

# Embedding layer
# Example:
#   Layer("emb_pos", "embedding", (SL,), (768,))
# Convention:
#   input_shape  = (num_rows, _)
#   output_shape = (num_columns,)
# Interpretation:
#   This corresponds to an output of size SL x 768.
#   The layer processes SL tokens, each represented by a vector of size 768.

# Residual layer
# Example:
#   Layer("residual0", "residual", (768,), (768,))
# Convention:
#   input_shape  = (num_rows, _)
#   output_shape = (num_columns,)
# Interpretation:
#   Residual layers perform an element-wise sum between tensors of the same
#   shape, typically over an activation matrix of size SL x 768.

# Fully connected (FC) layer mapped on CIM
# Example:
#   Layer(f"fc1{suf}", "fc", (768,), (2304,))
# Convention:
#   input_shape  = (num_rows, _)
#   output_shape = (num_columns,)
# Interpretation:
#   The FC layer corresponds to a weight matrix of size 768 x 2304 to be mapped
#   on CIM tiles.
#   It receives SL input vectors, each of size 768.
#   Per token, the operation is:
#       (1 x 768) x (768 x 2304) = (1 x 2304)

# Fully connected / GEMM layer mapped on digital accelerators
# Example:
#   Layer(f"gemm1{suf}", "gemm", (SL, 768), (SL, 2304))
# Convention:
#   input_shape  = (num_rows_input, num_columns_input)
#   output_shape = (num_rows_output, num_columns_output)
# Interpretation:
#   This layer performs a matrix multiplication on a digital accelerator:
#       (SL x 768) x (768 x 2304) = (SL x 2304)

# LayerNorm layer --> RMSNorm softmax
# Example:
#   Layer(f"layernorm1{suf}", "layernorm", (SL, 768), (SL, 768))
# Convention:
#   input_shape  = (num_rows, num_columns)
#   output_shape = (num_rows, num_columns)
# Interpretation:
#   Input and output have the same size.
#   LayerNorm is applied to an activation matrix of size SL x 768.

# Attention layer
# Example:
#   Layer(f"atten{suf}", "atten", (SL, 64), (SL, 64 * 12))
# Convention:
#   input_shape  = (num_rows_input, head_dim)
#   output_shape = (num_rows_output, head_dim * num_heads)
# Interpretation:
#   The input shape represents the size of Q, K, and V for one head:
#       (SL x head_dim)
#   The output shape represents the concatenated output of all heads:
#       (SL x hidden_dim)
#   where:
#       hidden_dim = head_dim * num_heads
#   Example:
#       head_dim = 64
#       num_heads = 12
#       hidden_dim = 768

# GeLU layer, element-wise 
# Example:
#   Layer(f"gelu{suf}", "gelu", (SL, 3072), (SL, 3072))
# Convention:
#   input_shape  = (num_rows, num_columns)
#   output_shape = (num_rows, num_columns)
# Interpretation:
#   Input and output have the same size.
#   GeLU is applied element-wise to an activation matrix of size SL x 3072.


class Layer:
    def __init__(self, name, layer_type, input_shape, output_shape, attention_type=None):
        self.name         = name
        self.type         = layer_type
        self.input_shape  = input_shape
        self.output_shape = output_shape
        self.attention_type = attention_type    # None, or "cross"

class Model:
    def __init__(self, name):
        self.name  = name
        self.graph = nx.DiGraph()

    def add_layer(self, layer, depends_on=None):
        self.graph.add_node(layer.name, layer=layer)
        if depends_on:
            for p in depends_on:
                self.graph.add_edge(p, layer.name)

    def get_graph(self):
        return self.graph

def get_model_dumpLayer(num_blocks=1, SL=1, nrow=512, ncol=512):
    dumpLayer = Model("dumpLayer")

    deps = []
    # --- Encoder Blocks ---
        # define all layers in block b
    for b in range(num_blocks):
        suf = f"_b{b}"
        fc1   = Layer(f"fc1{suf}", "fc", (nrow,), (ncol,))
        dumpLayer.add_layer(fc1,   [])

        dumpLayer.add_layer(fc1,   deps)
        deps = [fc1.name]

    return dumpLayer, SL, num_blocks


def get_model_dumpModel(num_blocks=1, SL=1):
    """Small illustrative test model (Fig. 6): each block is 6 FC layers --
    L0,L1,L2 (size 512x128, no dependency) feed L3,L4 (size 128x128), which
    feed L5 (size 640x128); L5 then feeds the next block's L0,L1,L2."""
    dumpModel = Model("dumpModel")

    deps = []
    # --- Encoder Blocks ---
    for b in range(num_blocks):
        suf = f"_b{b}"
        fc0   = Layer(f"fc0{suf}", "fc", (512,), (128,))
        fc1   = Layer(f"fc1{suf}", "fc", (512,), (128,))
        fc2   = Layer(f"fc2{suf}", "fc", (512,), (128,))
        fc3   = Layer(f"fc3{suf}", "fc", (128,), (128,))
        fc4   = Layer(f"fc4{suf}", "fc", (128,), (128,))
        fc5   = Layer(f"fc5{suf}", "fc", (640,), (128,))

        dumpModel.add_layer(fc0,   deps)
        dumpModel.add_layer(fc1,   deps)
        dumpModel.add_layer(fc2,   deps)
        dumpModel.add_layer(fc3,   [fc0.name, fc1.name, fc2.name])
        dumpModel.add_layer(fc4,   [fc0.name, fc1.name, fc2.name])
        dumpModel.add_layer(fc5,   [fc3.name, fc4.name])

        deps = [fc5.name]

    return dumpModel, SL, num_blocks


def get_model_mobileBERT(num_blocks=24, SL=128):
    mobileBERT = Model("MobileBERT")
    prev_fc15  = None

    # --- Embedding Layers ---
    # token, position, segment embeddings, followed by linear transform and dropout
    emb_word = Layer("emb_word", "embedding", (SL,), (128,))
    emb_pos  = Layer("emb_pos",  "embedding", (SL,), (512,))
    emb_seg  = Layer("emb_seg",  "embedding", (SL,), (512,))
    emb_lin  = Layer("emb_lin",  "fc",        (384,), (512,))   # transforms concat(word,pos,seg)

    # assemble embedding dependencies
    mobileBERT.add_layer(emb_word, [])
    mobileBERT.add_layer(emb_pos,  [])
    mobileBERT.add_layer(emb_seg,  [])
    mobileBERT.add_layer(emb_lin,  [emb_word.name, emb_pos.name, emb_seg.name])
 
    # the first block will depend on the embedding output
    prev_input = emb_lin.name

    # --- Encoder Blocks ---
    for b in range(num_blocks):
        suf = f"_b{b}"
        # define all layers in block b
        fc1   = Layer(f"fc1{suf}",   "fc",       (512,), (128,))
        fc2   = Layer(f"fc2{suf}",   "fc",       (512,), (128,))
        fc3   = Layer(f"fc3{suf}",   "fc",       (128,), (128,))
        fc4   = Layer(f"fc4{suf}",   "fc",       (128,), (128,))
        fc5   = Layer(f"fc5{suf}",   "fc",       (512,), (128,))
        atten = Layer(f"atten{suf}","atten",    (SL,32), (SL,32*4))
        fc6   = Layer(f"fc6{suf}",   "fc",       (128,), (128,))
        res1  = Layer(f"residual1{suf}","residual",(128,), (128,))
        fc7   = Layer(f"fc7{suf}",   "fc",       (128,), (512,))
        fc8   = Layer(f"fc8{suf}",   "fc",       (512,), (128,))
        res2  = Layer(f"residual2{suf}","residual",(128,), (128,))
        fc9   = Layer(f"fc9{suf}",   "fc",       (128,), (512,))
        fc10  = Layer(f"fc10{suf}",  "fc",       (512,), (128,))
        res3  = Layer(f"residual3{suf}","residual",(128,), (128,))
        fc11  = Layer(f"fc11{suf}",  "fc",       (128,), (512,))
        fc12  = Layer(f"fc12{suf}",  "fc",       (512,), (128,))
        res4  = Layer(f"residual4{suf}","residual",(128,), (128,))
        fc13  = Layer(f"fc13{suf}",  "fc",       (128,), (512,))
        fc14  = Layer(f"fc14{suf}",  "fc",       (512,), (128,))
        res5  = Layer(f"residual5{suf}","residual",(128,), (128,))
        fc15  = Layer(f"fc15{suf}",  "fc",       (128,), (512,))

        # build dependency lists chaining from prev_input
        deps_fc1   = [prev_input]
        deps_fc2   = deps_fc1.copy()
        deps_fc3   = [fc2.name] 
        deps_fc4   = [fc2.name]
        deps_fc5   = deps_fc1.copy()
        deps_atten = [fc3.name, fc4.name, fc5.name]
        deps_fc6   = [atten.name]
        deps_res1  = [fc1.name, fc6.name]
        deps_fc7   = [res1.name]
        deps_fc8   = [fc7.name]
        deps_res2  = [res1.name, fc8.name]
        deps_fc9   = [res2.name]
        deps_fc10  = [fc9.name]
        deps_res3  = [res2.name, fc10.name]
        deps_fc11  = [res3.name]
        deps_fc12  = [fc11.name]
        deps_res4  = [res3.name, fc12.name]
        deps_fc13  = [res4.name]
        deps_fc14  = [fc13.name]
        deps_res5  = [res4.name, fc14.name]
        deps_fc15  = [res5.name]

        # add to graph
        mobileBERT.add_layer(fc1,   deps_fc1)
        mobileBERT.add_layer(fc2,   deps_fc2)
        mobileBERT.add_layer(fc3,   deps_fc3)
        mobileBERT.add_layer(fc4,   deps_fc4)
        mobileBERT.add_layer(fc5,   deps_fc5)
        mobileBERT.add_layer(atten, deps_atten)
        mobileBERT.add_layer(fc6,   deps_fc6)
        mobileBERT.add_layer(res1,  deps_res1)
        mobileBERT.add_layer(fc7,   deps_fc7)
        mobileBERT.add_layer(fc8,   deps_fc8)
        mobileBERT.add_layer(res2,  deps_res2)
        mobileBERT.add_layer(fc9,   deps_fc9)
        mobileBERT.add_layer(fc10,  deps_fc10)
        mobileBERT.add_layer(res3,  deps_res3)
        mobileBERT.add_layer(fc11,  deps_fc11)
        mobileBERT.add_layer(fc12,  deps_fc12)
        mobileBERT.add_layer(res4,  deps_res4)
        mobileBERT.add_layer(fc13,  deps_fc13)
        mobileBERT.add_layer(fc14,  deps_fc14)
        mobileBERT.add_layer(res5,  deps_res5)
        mobileBERT.add_layer(fc15,  deps_fc15)

        # prepare for next block
        prev_input = fc15.name

    return mobileBERT, SL, num_blocks


def get_model_BERT_B(num_blocks=12, SL=128):
    BERT_B = Model("BERT_B")

    # --- Embedding Layers ---
    # token, position, segment embeddings, followed by linear transform and dropout
    emb_word = Layer("emb_word", "embedding", (SL,), (768,))
    emb_pos  = Layer("emb_pos",  "embedding", (SL,), (768,))
    emb_seg  = Layer("emb_seg",  "embedding", (SL,), (768,))
    #emb_lin  = Layer("emb_lin",  "fc",        (384,), (512,))   # transforms concat(word,pos,seg)
    emb_layernorm   = Layer(f"layernorm0",  "layernorm", (SL,768), (SL,768))

    # assemble embedding dependencies
    BERT_B.add_layer(emb_word, [])
    BERT_B.add_layer(emb_pos,  [])
    BERT_B.add_layer(emb_seg,  [])
    BERT_B.add_layer(emb_layernorm,  [emb_word.name, emb_pos.name, emb_seg.name])
 
    # the first block will depend on the embedding output
    prev_input = emb_layernorm.name

    # --- Encoder Blocks ---
    for b in range(num_blocks):
        suf = f"_b{b}"
        # define all layers in block b
        fc1   = Layer(f"fc1{suf}",   "fc",       (768,), (768,))
        fc2   = Layer(f"fc2{suf}",   "fc",       (768,), (768,))
        fc3   = Layer(f"fc3{suf}",   "fc",       (768,), (768,))
        atten = Layer(f"atten{suf}","atten",    (SL,64), (SL,64*12))
        fc4   = Layer(f"fc4{suf}",   "fc",       (768,), (768,))
        res1  = Layer(f"residual1{suf}","residual",(768,), (768,))
        layernorm1   = Layer(f"layernorm1{suf}",  "layernorm", (SL,768), (SL,768))
        fc5   = Layer(f"fc5{suf}",   "fc",       (768,), (3072,))
        gelu   = Layer(f"gelu{suf}",  "gelu", (SL,3072), (SL,3072))
        fc6   = Layer(f"fc6{suf}",   "fc",       (3072,), (768,))
        res2  = Layer(f"residual2{suf}","residual",(768,), (768,))
        layernorm2   = Layer(f"layernorm2{suf}",  "layernorm", (SL,768), (SL,768))

        # build dependency lists chaining from prev_input
        deps_fc1   = [prev_input]
        deps_fc2   = deps_fc1.copy()
        deps_fc3   = deps_fc1.copy()
        deps_atten = [fc1.name, fc2.name, fc3.name]
        deps_fc4   = [atten.name]
        deps_res1  = [fc4.name, prev_input]
        deps_layernorm1   = [res1.name]
        deps_fc5   = [layernorm1.name]
        deps_gelu   = [fc5.name]
        deps_fc6   = [gelu.name]
        deps_res2  = [fc6.name, layernorm1.name]
        deps_layernorm2   = [res2.name]

        # add to graph
        BERT_B.add_layer(fc1,   deps_fc1)
        BERT_B.add_layer(fc2,   deps_fc2)
        BERT_B.add_layer(fc3,   deps_fc3)
        BERT_B.add_layer(atten, deps_atten)
        BERT_B.add_layer(fc4,   deps_fc4)
        BERT_B.add_layer(res1,  deps_res1)
        BERT_B.add_layer(layernorm1,   deps_layernorm1)
        BERT_B.add_layer(fc5,   deps_fc5)
        BERT_B.add_layer(gelu,   deps_gelu)
        BERT_B.add_layer(fc6,   deps_fc6)
        BERT_B.add_layer(res2,  deps_res2)
        BERT_B.add_layer(layernorm2,   deps_layernorm2)

        # prepare for next block
        prev_input = layernorm2.name

    return BERT_B, SL, num_blocks

def get_model_ALBERT_B(num_blocks=12, SL=128):
    ALBERT_B = Model("ALBERT_B")

    # --- Embedding Layers ---
    # token, position, segment embeddings, followed by linear transform and dropout
    emb_word = Layer("emb_word", "embedding", (SL,), (768,))
    emb_pos  = Layer("emb_pos",  "embedding", (SL,), (768,))
    emb_seg  = Layer("emb_seg",  "embedding", (SL,), (768,))
    #emb_lin  = Layer("emb_lin",  "fc",        (384,), (512,))   # transforms concat(word,pos,seg)
    emb_layernorm   = Layer(f"layernorm0",  "layernorm", (SL,768), (SL,768))

    # assemble embedding dependencies
    ALBERT_B.add_layer(emb_word, [])
    ALBERT_B.add_layer(emb_pos,  [])
    ALBERT_B.add_layer(emb_seg,  [])
    ALBERT_B.add_layer(emb_layernorm,  [emb_word.name, emb_pos.name, emb_seg.name])
 
    # the first block will depend on the embedding output
    prev_input = emb_layernorm.name

    # --- Encoder Blocks ---
    for b in range(num_blocks):
        suf = f"_b{b}"
        # define all layers in block b
        fc1   = Layer(f"fc1{suf}",   "fc",       (768,), (768,))
        # fc1   = Layer(f"gemm1{suf}",   "gemm",       (SL,768), (SL,768))
        fc2   = Layer(f"fc2{suf}",   "fc",       (768,), (768,))
        # fc2   = Layer(f"gemm2{suf}",   "gemm",       (SL,768), (SL,768))
        fc3   = Layer(f"fc3{suf}",   "fc",       (768,), (768,))
        # fc3   = Layer(f"gemm3{suf}",   "gemm",       (SL,768), (SL,768))
        atten = Layer(f"atten{suf}","atten",    (SL,64), (SL,64*12))
        fc4   = Layer(f"fc4{suf}",   "fc",       (768,), (768,))
        # fc4   = Layer(f"gemm4{suf}",   "gemm",       (SL,768), (SL,768))
        res1  = Layer(f"residual1{suf}","residual",(768,), (768,))
        layernorm1   = Layer(f"layernorm1{suf}",  "layernorm", (SL,768), (SL,768))
        # fc5   = Layer(f"fc5{suf}",   "fc",       (768,), (3072,))
        fc5   = Layer(f"gemm1{suf}",   "gemm",       (SL,768), (SL,3072))
        gelu   = Layer(f"gelu{suf}",  "gelu", (SL,3072), (SL,3072))
        # fc6   = Layer(f"fc6{suf}",   "fc",       (3072,), (768,))
        fc6   = Layer(f"gemm2{suf}",   "gemm",       (SL,3072), (SL,768))
        res2  = Layer(f"residual2{suf}","residual",(768,), (768,))
        layernorm2   = Layer(f"layernorm2{suf}",  "layernorm", (SL,768), (SL,768))

        # build dependency lists chaining from prev_input
        deps_fc1   = [prev_input]
        deps_fc2   = deps_fc1.copy()
        deps_fc3   = deps_fc1.copy()
        deps_atten = [fc1.name, fc2.name, fc3.name]
        deps_fc4   = [atten.name]
        deps_res1  = [fc4.name, prev_input]
        deps_layernorm1   = [res1.name]
        deps_fc5   = [layernorm1.name]
        deps_gelu   = [fc5.name]
        deps_fc6   = [gelu.name]
        deps_res2  = [fc6.name, layernorm1.name]
        deps_layernorm2   = [res2.name]

        # add to graph
        ALBERT_B.add_layer(fc1,   deps_fc1)
        ALBERT_B.add_layer(fc2,   deps_fc2)
        ALBERT_B.add_layer(fc3,   deps_fc3)
        ALBERT_B.add_layer(atten, deps_atten)
        ALBERT_B.add_layer(fc4,   deps_fc4)
        ALBERT_B.add_layer(res1,  deps_res1)
        ALBERT_B.add_layer(layernorm1,   deps_layernorm1)
        ALBERT_B.add_layer(fc5,   deps_fc5)
        ALBERT_B.add_layer(gelu,   deps_gelu)
        ALBERT_B.add_layer(fc6,   deps_fc6)
        ALBERT_B.add_layer(res2,  deps_res2)
        ALBERT_B.add_layer(layernorm2,   deps_layernorm2)

        # prepare for next block
        prev_input = layernorm2.name

    return ALBERT_B, SL, num_blocks


def get_model_BERT_L(num_blocks=24, SL=128):
    BERT_L = Model("BERT_L")

    # --- Embedding Layers ---
    # token, position, segment embeddings, followed by linear transform and dropout
    emb_word = Layer("emb_word", "embedding", (SL,), (1024,))
    emb_pos  = Layer("emb_pos",  "embedding", (SL,), (1024,))
    emb_seg  = Layer("emb_seg",  "embedding", (SL,), (1024,))
    #emb_lin  = Layer("emb_lin",  "fc",        (384,), (512,))   # transforms concat(word,pos,seg)
    emb_layernorm   = Layer(f"layernorm0",  "layernorm", (SL,1024), (SL,1024))

    # assemble embedding dependencies
    BERT_L.add_layer(emb_word, [])
    BERT_L.add_layer(emb_pos,  [])
    BERT_L.add_layer(emb_seg,  [])
    BERT_L.add_layer(emb_layernorm,  [emb_word.name, emb_pos.name, emb_seg.name])
 
    # the first block will depend on the embedding output
    prev_input = emb_layernorm.name

    # --- Encoder Blocks ---
    for b in range(num_blocks):
        suf = f"_b{b}"
        # define all layers in block b
        fc1   = Layer(f"fc1{suf}",   "fc",       (1024,), (1024,))
        fc2   = Layer(f"fc2{suf}",   "fc",       (1024,), (1024,))
        fc3   = Layer(f"fc3{suf}",   "fc",       (1024,), (1024,))
        atten = Layer(f"atten{suf}","atten",    (SL,64), (SL,64*16))
        fc4   = Layer(f"fc4{suf}",   "fc",       (1024,), (1024,))
        res1  = Layer(f"residual1{suf}","residual",(1024,), (1024,))
        layernorm1   = Layer(f"layernorm1{suf}",  "layernorm", (SL,1024), (SL,1024))
        fc5   = Layer(f"fc5{suf}",   "fc",       (1024,), (4096,))
        gelu   = Layer(f"gelu{suf}",  "gelu", (SL,4096), (SL,4096))
        fc6   = Layer(f"fc6{suf}",   "fc",       (4096,), (1024,))
        res2  = Layer(f"residual2{suf}","residual",(1024,), (1024,))
        layernorm2   = Layer(f"layernorm2{suf}",  "layernorm", (SL,1024), (SL,1024))

        # build dependency lists chaining from prev_input
        deps_fc1   = [prev_input]
        deps_fc2   = deps_fc1.copy()
        deps_fc3   = deps_fc1.copy()
        deps_atten = [fc1.name, fc2.name, fc3.name]
        deps_fc4   = [atten.name]
        deps_res1  = [fc4.name, prev_input]
        deps_layernorm1   = [res1.name]
        deps_fc5   = [layernorm1.name]
        deps_gelu   = [fc5.name]
        deps_fc6   = [gelu.name]
        deps_res2  = [fc6.name, layernorm1.name]
        deps_layernorm2   = [res2.name]

        # add to graph
        BERT_L.add_layer(fc1,   deps_fc1)
        BERT_L.add_layer(fc2,   deps_fc2)
        BERT_L.add_layer(fc3,   deps_fc3)
        BERT_L.add_layer(atten, deps_atten)
        BERT_L.add_layer(fc4,   deps_fc4)
        BERT_L.add_layer(res1,  deps_res1)
        BERT_L.add_layer(layernorm1,   deps_layernorm1)
        BERT_L.add_layer(fc5,   deps_fc5)
        BERT_L.add_layer(gelu,   deps_gelu)
        BERT_L.add_layer(fc6,   deps_fc6)
        BERT_L.add_layer(res2,  deps_res2)
        BERT_L.add_layer(layernorm2,   deps_layernorm2)

        # prepare for next block
        prev_input = layernorm2.name

    return BERT_L, SL, num_blocks



def get_model_NanoGPT(num_blocks=12, SL=512):
    NanoGPT = Model("NanoGPT")

    # --- Embedding Layers ---
    # token, position, segment embeddings, followed by linear transform and dropout
    emb_word = Layer("emb_word", "embedding", (SL,), (768,))
    emb_pos  = Layer("emb_pos",  "embedding", (SL,), (768,))
    elemwise_sum = Layer(f"residual0","residual",(768,), (768,))
    
    # assemble embedding dependencies
    NanoGPT.add_layer(emb_word, [])
    NanoGPT.add_layer(emb_pos,  [])
    NanoGPT.add_layer(elemwise_sum,  [emb_word.name, emb_pos.name])
 
    # the first block will depend on the embedding output
    prev_input = elemwise_sum.name
    # --- Encoder Blocks ---
    for b in range(num_blocks):
        suf = f"_b{b}"
        # define all layers in block b
        layernorm1   = Layer(f"layernorm1{suf}",  "layernorm", (SL,768), (SL,768))
        fc1   = Layer(f"fc1{suf}",   "fc",       (768,), (2304,))
        atten = Layer(f"atten{suf}","atten",    (SL,64), (SL,64*12))
        fc2   = Layer(f"fc2{suf}",   "fc",       (768,), (768,))
        res1  = Layer(f"residual1{suf}","residual",(768,), (768,))
        layernorm2   = Layer(f"layernorm2{suf}",  "layernorm", (SL,768), (SL,768))
        fc3   = Layer(f"fc3{suf}",   "fc",       (768,), (3072,))
        gelu   = Layer(f"gelu{suf}",  "gelu", (SL,3072), (SL,3072))
        fc4   = Layer(f"fc4{suf}",   "fc",       (3072,), (768,))
        res2  = Layer(f"residual2{suf}","residual",(768,), (768,))        
        # layernorm3   = Layer(f"layernorm3{suf}",  "layernorm", (SL,768), (SL,768))

        # build dependency lists chaining from prev_input
        deps_layernorm1   = [prev_input]
        deps_fc1   = [layernorm1.name]
        deps_atten = [fc1.name]
        deps_fc2   = [atten.name]
        deps_res1  = [fc2.name, prev_input]
        deps_layernorm2   = [res1.name]
        deps_fc3   = [layernorm2.name]
        deps_gelu   = [fc3.name]
        deps_fc4   = [gelu.name]
        deps_res2  = [fc4.name, res1.name]
        # deps_layernorm3   = [res2.name]

        # add to graph
        NanoGPT.add_layer(layernorm1,   deps_layernorm1)
        NanoGPT.add_layer(fc1,   deps_fc1)
        NanoGPT.add_layer(atten, deps_atten)
        NanoGPT.add_layer(fc2,   deps_fc2)
        NanoGPT.add_layer(res1,  deps_res1)
        NanoGPT.add_layer(layernorm2,   deps_layernorm2)
        NanoGPT.add_layer(fc3,   deps_fc3)
        NanoGPT.add_layer(gelu,   deps_gelu)
        NanoGPT.add_layer(fc4,   deps_fc4)
        NanoGPT.add_layer(res2,  deps_res2)
        # NanoGPT.add_layer(layernorm3,   deps_layernorm3)

        # prepare for next block
        prev_input = res2.name

    # last_layer   = Layer(f"fc5_b{0}",   "fc",       (768,), (50257,))
    layernorm3   = Layer(f"layernorm3_b{0}",  "layernorm", (SL,768), (SL,768))
    last_layer   = Layer(f"fc5_b{0}",   "fc",       (768,), (50257,))
    
    deps_layernorm3   = [res2.name]
    deps_last_layer = [layernorm3.name]

    NanoGPT.add_layer(layernorm3,   deps_layernorm3)
    NanoGPT.add_layer(last_layer,   deps_last_layer)

    return NanoGPT, SL, num_blocks

def get_model_GPT2_small(num_blocks=12, SL=512):
    GPT2s = Model("NanoGPT")

    # --- Embedding Layers ---
    # token, position, segment embeddings, followed by linear transform and dropout
    emb_word = Layer("emb_word", "embedding", (SL,), (768,))
    emb_pos  = Layer("emb_pos",  "embedding", (SL,), (768,))
    elemwise_sum = Layer(f"residual0","residual",(768,), (768,))
    
    # assemble embedding dependencies
    GPT2s.add_layer(emb_word, [])
    GPT2s.add_layer(emb_pos,  [])
    GPT2s.add_layer(elemwise_sum,  [emb_word.name, emb_pos.name])
 
    # the first block will depend on the embedding output
    prev_input = elemwise_sum.name
    # --- Encoder Blocks ---
    for b in range(num_blocks):
        suf = f"_b{b}"
        # define all layers in block b
        layernorm1   = Layer(f"layernorm1{suf}",  "layernorm", (SL,768), (SL,768))
        fc1   = Layer(f"fc1{suf}",   "fc",       (768,), (2304,))
        atten = Layer(f"atten{suf}","atten",    (SL,64), (SL,64*12))
        fc2   = Layer(f"fc2{suf}",   "fc",       (768,), (768,))
        res1  = Layer(f"residual1{suf}","residual",(768,), (768,))
        layernorm2   = Layer(f"layernorm2{suf}",  "layernorm", (SL,768), (SL,768))
        fc3   = Layer(f"fc3{suf}",   "fc",       (768,), (3072,))
        gelu   = Layer(f"gelu{suf}",  "gelu", (SL,3072), (SL,3072))
        fc4   = Layer(f"fc4{suf}",   "fc",       (3072,), (768,))
        res2  = Layer(f"residual2{suf}","residual",(768,), (768,))        
        # layernorm3   = Layer(f"layernorm3{suf}",  "layernorm", (SL,768), (SL,768))

        # build dependency lists chaining from prev_input
        deps_layernorm1   = [prev_input]
        deps_fc1   = [layernorm1.name]
        deps_atten = [fc1.name]
        deps_fc2   = [atten.name]
        deps_res1  = [fc2.name, prev_input]
        deps_layernorm2   = [res1.name]
        deps_fc3   = [layernorm2.name]
        deps_gelu   = [fc3.name]
        deps_fc4   = [gelu.name]
        deps_res2  = [fc4.name, res1.name]
        # deps_layernorm3   = [res2.name]

        # add to graph
        GPT2s.add_layer(layernorm1,   deps_layernorm1)
        GPT2s.add_layer(fc1,   deps_fc1)
        GPT2s.add_layer(atten, deps_atten)
        GPT2s.add_layer(fc2,   deps_fc2)
        GPT2s.add_layer(res1,  deps_res1)
        GPT2s.add_layer(layernorm2,   deps_layernorm2)
        GPT2s.add_layer(fc3,   deps_fc3)
        GPT2s.add_layer(gelu,   deps_gelu)
        GPT2s.add_layer(fc4,   deps_fc4)
        GPT2s.add_layer(res2,  deps_res2)

        # prepare for next block
        prev_input = res2.name

    # last_layer   = Layer(f"fc5_b{0}",   "fc",       (768,), (50257,))
    layernorm3   = Layer(f"layernorm3_b{0}",  "layernorm", (SL,768), (SL,768))
    last_layer   = Layer(f"fc5_b{0}",   "fc",       (768,), (50257,))
    
    deps_layernorm3   = [res2.name]
    deps_last_layer = [layernorm3.name]

    GPT2s.add_layer(layernorm3,   deps_layernorm3)
    GPT2s.add_layer(last_layer,   deps_last_layer)


    return GPT2s, SL, num_blocks


def get_model_T5_small_encoder(num_blocks=6, SL=128, d_model=512, d_ff=2048, num_heads=8, d_kv=64):
    """
    T5-small encoder graph:
      - token embedding
      - for each block:
          layernorm -> self-attn (q,k,v,o + relpos bias) -> residual
          layernorm -> FFN (wi, wo + act) -> residual
      - final layernorm
    """
    T5_ENC = Model("T5_small_encoder")

    # --- Embedding ---
    # (Shared embedding exists in T5ForConditionalGeneration; encoder has embed_tokens too.
    # For graph purposes, keep one embedding node that produces (SL, d_model).)
    emb_tok = Layer("emb_tok", "embedding", (SL,), (d_model,))
    T5_ENC.add_layer(emb_tok, [])

    prev_input = emb_tok.name

    # --- Encoder blocks ---
    for b in range(num_blocks):
        suf = f"_b{b}"

        # Pre-norm for self-attention
        ln_sa = Layer(f"layernorm1{suf}", "layernorm", (SL, d_model), (SL, d_model))

        # Self-attention projections
        q = Layer(f"fc1{suf}", "fc", (d_model,), (d_model,))
        k = Layer(f"fc2{suf}", "fc", (d_model,), (d_model,))
        v = Layer(f"fc3{suf}", "fc", (d_model,), (d_model,))
        # Attention core (scores/softmax/weighted sum). Use your "atten" layer type.
        # Shapes here are schematic like your BERT example; adjust to your internal convention.
        attn = Layer(f"atten{suf}", "atten", (SL, d_kv), (SL, d_kv * num_heads))
        o = Layer(f"fc4{suf}", "fc", (d_model,), (d_model,))

        # Residual add after attention
        res_sa = Layer(f"residual1{suf}", "residual", (d_model,), (d_model,))

        # Pre-norm for FFN
        ln_ff = Layer(f"layernorm2{suf}", "layernorm", (SL, d_model), (SL, d_model))

        # FFN: wi: d_model -> d_ff, act, wo: d_ff -> d_model
        wi = Layer(f"fc5{suf}", "fc", (d_model,), (d_ff,)) # ReLU 
        wo = Layer(f"fc6{suf}", "fc", (d_ff,), (d_model,))

        # Residual add after FFN
        res_ff = Layer(f"residual2{suf}", "residual", (d_model, ), (d_model,))

        #ln_ff_post = Layer(f"layernorm3{suf}", "layernorm", (SL, d_model), (SL, d_model))


        # build dependency lists chaining from prev_input
        deps_ln_sa   = [prev_input]
        deps_q   = [ln_sa.name]
        deps_k   = [ln_sa.name]
        deps_v   = [ln_sa.name]
        deps_atten = [q.name, k.name, v.name]
        deps_o   = [attn.name]
        deps_res1  = [o.name, prev_input]
        deps_layernorm2   = [res_sa.name]
        deps_wi   = [ln_ff.name]
        deps_wo   = [wi.name]
        deps_res2   = [wo.name, res_sa.name]

        # --- Dependencies ---
        # LN takes prev_input
        T5_ENC.add_layer(ln_sa, deps_ln_sa)

        # q,k,v depend on ln_sa output
        T5_ENC.add_layer(q,deps_q )
        T5_ENC.add_layer(k, deps_k)
        T5_ENC.add_layer(v, deps_v)
        # attention depends on q,k,v + relpos bias table
        T5_ENC.add_layer(attn, deps_atten)
        # output projection depends on attention output
        T5_ENC.add_layer(o, deps_o)
        # residual add: (o + prev_input)
        T5_ENC.add_layer(res_sa, deps_res1)
        # FFN LN depends on residual from attention
        T5_ENC.add_layer(ln_ff,deps_layernorm2)

        # FFN path
        T5_ENC.add_layer(wi, deps_wi)
        T5_ENC.add_layer(wo, deps_wo)
        # residual add: (wo + res_sa)
        T5_ENC.add_layer(res_ff, deps_res2)

        prev_input = res_ff.name

    # Final layer norm
    ln_final = Layer(f"layernorm4_b{0}", "layernorm", (SL, d_model), (SL, d_model))
    deps_layernorm4   = [res_ff.name]
    T5_ENC.add_layer(ln_final, deps_layernorm4)

    return T5_ENC, SL, num_blocks


def get_model_T5_small_decoder(num_blocks=6, SL=1, d_model=512, d_ff=2048, num_heads=8, d_kv=64, SL_cross=128):
    """
    T5-small decoder graph:
      - token embedding (decoder input length typically differs from encoder; we reuse SL for simplicity)
      - for each block:
          layernorm -> masked self-attn -> residual
          layernorm -> cross-attn (Q from decoder, K/V from encoder) -> residual
          layernorm -> FFN -> residual
      - final layernorm
      - lm_head projection to vocab
    """
    T5_DEC = Model("T5_small_decoder")

    # Decoder embedding
    emb_tok = Layer("emb_tok", "embedding", (SL,), (d_model,))
    T5_DEC.add_layer(emb_tok, [])

    prev_input = emb_tok.name

    # IMPORTANT: decoder cross-attn needs encoder output as an input.
    # Represent this as a special "input" layer that upstream code can connect to encoder final output.
    # enc_out = Layer("enc_out", "input", (SL,), (d_model, )) #currently we can fetch from DRAM.
    # T5_DEC.add_layer(enc_out, [])
    enc_out = Layer("emb_enc_out", "embedding", (SL_cross,), (d_model, )) #currently we can fetch from DRAM.
    T5_DEC.add_layer(enc_out, [])

    for b in range(num_blocks):
        suf = f"_b{b}"

        ln_sa = Layer(f"layernorm1{suf}", "layernorm", (SL, d_model), (SL, d_model))
        # --- Masked self-attention sublayer ---
        q = Layer(f"fc1{suf}", "fc", (d_model,), (d_model,))
        k = Layer(f"fc2{suf}", "fc", (d_model,), (d_model,))
        v = Layer(f"fc3{suf}", "fc", (d_model,), (d_model,))
        attn = Layer(f"atten0{suf}", "atten", (SL, d_kv), (SL, d_kv * num_heads)) #masked self-attention, affected by prefill param
        o = Layer(f"fc4{suf}", "fc", (d_model,), (d_model,))
        res_sa = Layer(f"residual1{suf}", "residual", (SL, d_model), (SL, d_model))

        deps_ln_sa = [prev_input]
        deps_q = [ln_sa.name]
        deps_k = [ln_sa.name]
        deps_v = [ln_sa.name]
        deps_atten = [q.name, k.name, v.name] # + relpos bias table (not shown as explicit node here)
        deps_o = [attn.name]
        deps_res_sa = [o.name, prev_input]

        T5_DEC.add_layer(ln_sa, deps_ln_sa)
        T5_DEC.add_layer(q, deps_q)
        T5_DEC.add_layer(k, deps_k)
        T5_DEC.add_layer(v, deps_v)
        T5_DEC.add_layer(attn, deps_atten)
        T5_DEC.add_layer(o, deps_o)
        T5_DEC.add_layer(res_sa, deps_res_sa)

        # --- Cross-attention sublayer (EncDecAttention) ---
        ln_ca = Layer(f"layernorm2{suf}", "layernorm", (SL, d_model), (SL, d_model))
        qx = Layer(f"fc5{suf}", "fc", (d_model,), (d_model,))
        kx = Layer(f"fc6{suf}", "fc", (d_model,), (d_model,), attention_type="cross")
        vx = Layer(f"fc7{suf}", "fc", (d_model,), (d_model,), attention_type="cross")
        cross = Layer(f"atten1{suf}", "atten", (SL, d_kv), (SL, d_kv * num_heads), attention_type="cross") # cross-attention;
        ox = Layer(f"fc8{suf}", "fc", (d_model,), (d_model,))
        res_ca = Layer(f"residual2{suf}", "residual", (SL, d_model), (SL, d_model))

        deps_ln_ca = [res_sa.name]
        deps_qx = [ln_ca.name]
        deps_kx = [enc_out.name]
        deps_vx = [enc_out.name]
        deps_cross = [qx.name, kx.name, vx.name]
        deps_ox = [cross.name] 
        deps_res_ca = [ox.name, res_sa.name]

        # LN for cross-attn depends on output of self-attn residual
        T5_DEC.add_layer(ln_ca, deps_ln_ca)

        # Q comes from decoder stream (ln_ca)
        T5_DEC.add_layer(qx, deps_qx)
        # K/V come from encoder output (enc_out)
        T5_DEC.add_layer(kx, deps_kx) #enc_out
        T5_DEC.add_layer(vx, deps_vx) #enc_out

        T5_DEC.add_layer(cross, deps_cross)
        T5_DEC.add_layer(ox, deps_ox)

        # residual add: (ox + res_sa)
        T5_DEC.add_layer(res_ca, deps_res_ca)

        # --- FFN sublayer ---
        ln_ff = Layer(f"layernorm3{suf}", "layernorm", (SL, d_model), (SL, d_model))
        wi = Layer(f"fc9{suf}", "fc", (d_model,), (d_ff,)) #ReLU
        wo = Layer(f"fc10{suf}", "fc", (d_ff,), (d_model,))
        res_ff = Layer(f"residual3{suf}", "residual", (SL, d_model), (SL, d_model))

        deps_ln_ff = [res_ca.name]
        deps_wi = [ln_ff.name]
        deps_wo = [wi.name]
        deps_res_ff = [wo.name, res_ca.name]

        T5_DEC.add_layer(ln_ff, deps_ln_ff)
        T5_DEC.add_layer(wi, deps_wi)
        T5_DEC.add_layer(wo, deps_wo)
        T5_DEC.add_layer(res_ff, deps_res_ff)

        prev_input = res_ff.name

    # Final layer norm + LM head
    ln_final = Layer(f"layernorm4_b{0}", "layernorm", (SL, d_model), (SL, d_model))
    lm_head = Layer(f"fc11_b{0}", "fc", (d_model,), (32128,))  # vocab_size=32128

    T5_DEC.add_layer(ln_final, [prev_input])
    T5_DEC.add_layer(lm_head, [ln_final.name])

    return T5_DEC, SL, num_blocks, SL_cross