"""
Manifold Muon optimizer for nanoGPT

Ported from successful CIFAR-10 experiments:
- 65.21% test accuracy (no skip, rank=160)
- 60.96% test accuracy (with skip, rank=160, G_{0.0, -1.0, r})
"""

from .grassmann_muon import grassmann_muon, grassmann_muon_update
from .grassmann_ops import (
    tangent_operator_LX,
    retract_to_grassmann,
    initialize_on_grassmann,
)
from .msign import msign
from .block_decomposition import BlockDecomposedLinear, FullBlockDecomposedLinear

__all__ = [
    'grassmann_muon',
    'grassmann_muon_update',
    'tangent_operator_LX',
    'retract_to_grassmann',
    'initialize_on_grassmann',
    'msign',
    'BlockDecomposedLinear',
    'FullBlockDecomposedLinear',
]
