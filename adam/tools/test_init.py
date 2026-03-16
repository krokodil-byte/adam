"""ADAM — quick initialization test."""
import os
from adam.paths import ROOT as _ROOT, setup; setup()

from adam.loaders.gguf import GGUFLoader
from adam.models.engine import ADAMEngine, ModelConfig
import adamah as adamah_mod

MODEL_PATH = os.path.join(_ROOT, "gemma3-1b.gguf")
print("Loading GGUF...")
loader = GGUFLoader(MODEL_PATH)
loader.load(verbose=True)

print("Init GPU...")
gpu = adamah_mod.init()
print("GPU initialized")

cfg = ModelConfig.from_gguf_metadata(loader.metadata, verbose=True)
print("Init Engine...")
engine = ADAMEngine(gpu, cfg, loader.tensors,
                    raw_blocks=loader.raw_blocks,
                    tensor_types=loader.tensor_types,
                    adamah_mod=adamah_mod, verbose=True)
print("Engine initialized")
