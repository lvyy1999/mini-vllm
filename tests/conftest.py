import os
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType


os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Unit tests import source submodules directly instead of eagerly constructing the
# full CUDA runtime exposed by minivllm.__init__.
PACKAGE_DIR = SRC_DIR / "minivllm"
if "minivllm" not in sys.modules:
    package = ModuleType("minivllm")
    package.__file__ = str(PACKAGE_DIR / "__init__.py")
    package.__package__ = "minivllm"
    package.__path__ = [str(PACKAGE_DIR)]
    package.__spec__ = ModuleSpec("minivllm", loader=None, is_package=True)
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules["minivllm"] = package
