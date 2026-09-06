"""Standalone TextGrad validation experiment package.

This package deliberately has no imports from the main project pipeline.
"""
from __future__ import annotations

import os

# ``textgrad`` imports ``litellm``, which by default tries to fetch a remote
# model-cost map at import time and warns on failure. The offline experiment
# always uses the bundled local map, so disable the network fetch up front
# (must be set before ``litellm`` is imported anywhere in the process).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

