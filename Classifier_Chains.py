import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).with_name("Classifier Chains.py")
_SPEC = importlib.util.spec_from_file_location("classifier_chains_legacy", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

GloveClassifierChainModel = _MODULE.GloveClassifierChainModel
BertClassifierChainModel = _MODULE.BertClassifierChainModel
CreateGloveModel = _MODULE.CreateGloveModel
CreateBertModel = _MODULE.CreateBertModel
CreateModel = _MODULE.CreateModel

__all__ = [
    "GloveClassifierChainModel",
    "BertClassifierChainModel",
    "CreateGloveModel",
    "CreateBertModel",
    "CreateModel",
]
