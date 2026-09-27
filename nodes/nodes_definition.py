#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#

from utils.chip_utils import dechip
class AcceleratorInstance:
    def __init__(self, name, type_tag, supported_ops, cost_model, power_model=None, chip_id: int=0):
        self.name = name              # e.g., "ConvAccel[0]"
        self.type_tag = type_tag      # e.g., "ConvAccel"
        self.supported_ops = supported_ops
        self.cost_model = cost_model
        self.power_model = power_model or (lambda layer: 0.0)
        self.available_time = 0       # When this unit is next free
        self.schedule = []            # List of (layer_name, start, duration)
        self.chip_id          = chip_id

    def can_run(self, layer):
        return layer.type in self.supported_ops

    def estimate_cost(self, layer, **kwargs): # pipelining=None, parallelism=None, inter_layer_chunk_size=None):
        return self.cost_model(layer, **kwargs)#pipelining=pipelining, parallelism=parallelism, inter_layer_chunk_size=inter_layer_chunk_size)
    
    def estimate_power(self, layer):
        return self.power_model(layer)
    
    # Unified type string without prefix
    @property
    def basename(self) -> str:
        return dechip(self.name)
