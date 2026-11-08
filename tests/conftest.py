"""
Test setup. The real embedding model and the LLM can't run in CI (no model download,
no API key), so the tests use small stand-ins for them and test everything else for real.
"""

import os
import sys
import types

# Make the project folder importable (ingest.py, rag.py, evaluate.py live there).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stand-in for sentence_transformers, only if the real library isn't installed.
if "sentence_transformers" not in sys.modules:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=object)
