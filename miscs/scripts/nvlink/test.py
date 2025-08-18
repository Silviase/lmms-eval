import sys
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
def safe_import(name):
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"{name}: {ver} (import OK)")
    except Exception as e:
        print(f"{name}: import FAILED -> {e.__class__.__name__}: {e}")
        sys.exit(1)

safe_import("xformers")
safe_import("flash_attn")
print("All good ✅")