import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).with_name("Binary Relevance.py")
_SPEC = importlib.util.spec_from_file_location("binary_relevance_legacy", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

create_model = _MODULE.create_model
BertBinaryRelevanceModel = _MODULE.BertBinaryRelevanceModel
CreateModel = _MODULE.CreateModel

__all__ = ["create_model", "BertBinaryRelevanceModel", "CreateModel"]
