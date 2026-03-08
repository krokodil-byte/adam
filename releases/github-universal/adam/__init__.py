"""
ADAM — Adaptive Dispatch Architecture for Models
=================================================
GGUF model loader and inference engine built on ADAMAH (Vulkan GPU).

Supports any transformer model in GGUF format.

Copyright (c) 2026 Samuele Scuglia
License: CC BY-NC 4.0 (see LICENSE)
"""

from adam.loaders.gguf import GGUFLoader
from adam.tokenizers.gguf_tok import GGUFTokenizer
from adam.models.engine import ADAMEngine, ModelConfig, GenerationConfig

__version__ = "1.0.0"
__all__ = [
    "GGUFLoader",
    "GGUFTokenizer",
    "ADAMEngine",
    "ModelConfig",
    "GenerationConfig",
]
