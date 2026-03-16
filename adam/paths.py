"""Centralized path setup for ADAM project."""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAMAH_DIR = os.path.join(ROOT, "adamah-MAIN")

def setup():
    """Add ROOT and ADAMAH_DIR to sys.path if not already present."""
    for p in [ROOT, ADAMAH_DIR]:
        if p not in sys.path:
            sys.path.insert(0, p)
