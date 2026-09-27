#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#

from nodes.cost_models import MemoryEstimator

class LayerSplitter:
    @staticmethod
    def split_sublayer_to_fit_tcdm(layer, tcdm_limit_kb=128):
        """
        Split the layer recursively across rows or columns until each sublayer fits in TCDM.
        """
        splits = [layer]
        final_splits = []

        while splits:
            current = splits.pop()
            mem_kb = MemoryEstimator.estimate_memory_kb(current)

            if mem_kb <= tcdm_limit_kb:
                final_splits.append(current)
                continue

            in_r, in_c = current.input_shape
            out_r, out_c = current.output_shape

            if layer.name.startswith("gemm"):
                if out_c >= 2:
                    new_c = out_c // 2
                    sub1 = type(current)(
                        name=f"{current.name}_fit_c0",
                        input_shape=(in_r, in_c),
                        output_shape=(out_r, new_c),
                        layer_type=current.type
                    )
                    sub2 = type(current)(
                        name=f"{current.name}_fit_c1",
                        input_shape=(in_r, in_c),
                        output_shape=(out_r, out_c - new_c),
                        layer_type=current.type
                    )
                    splits.extend([sub2, sub1])
                else:
                    raise RuntimeError(f"Cannot split layer {current.name} further to fit in TCDM.")
            else:
                if in_r >= 2:
                    new_r = in_r // 2
                    sub1 = type(current)(
                        name=f"{current.name}_split0_r{new_r}",
                        input_shape=(new_r, in_c),
                        output_shape=(new_r, out_c),
                        layer_type=current.type
                    )
                    sub2 = type(current)(
                        name=f"{current.name}_split1_r{new_r}",
                        input_shape=(in_r - new_r, in_c),
                        output_shape=(in_r - new_r, out_c),
                        layer_type=current.type
                    )
                    splits.extend([sub2, sub1])
                else:
                    raise RuntimeError(f"Cannot split layer {current.name} further to fit in TCDM.")

        return final_splits
    

    def split_attention_to_fit_tcdm(attention_layer, tcdm_capacity_kb):
        sl, head_d = attention_layer.input_shape
        _, hidden_size = attention_layer.output_shape

        num_chunks = 1
        sublayers = []

        while True:
            chunk_size = sl // num_chunks
            extra = sl % num_chunks

            if chunk_size == 0:
                raise RuntimeError(f"Cannot split attention layer {attention_layer.name} to fit in TCDM.")

            sublayers.clear()
            fits = True
            row_start = 0

            for i in range(num_chunks):
                current_chunk_size = chunk_size + (1 if i < extra else 0)

                sublayer = type(attention_layer)(
                    name=f"{attention_layer.name}_splitQ{current_chunk_size}_{i}",
                    input_shape=(current_chunk_size, head_d),
                    output_shape=(current_chunk_size, hidden_size),
                    layer_type="atten"
                )
                sublayer.parent_layer = attention_layer

                mem_kb = MemoryEstimator.estimate_memory_kb(sublayer, sl_full=sl)
                if mem_kb > tcdm_capacity_kb:
                    fits = False
                    break

                sublayers.append(sublayer)
                row_start += current_chunk_size

            if fits:
                break
            num_chunks += 1

        return sublayers
