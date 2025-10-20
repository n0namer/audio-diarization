try:
    import torio
    version = None
    for attr in ("__version__", "VERSION", "version"):
        if hasattr(torio, attr):
            version = getattr(torio, attr)
            break
    if version is None:
        try:
            import importlib.metadata
            version = importlib.metadata.version("torio")
        except Exception as e:
            version = f"not found via importlib.metadata: {e}"
    print("torio:", version)
    print("torio path:", torio.__file__)
except ImportError as e:
    print("torio: not installed:", e)
try:
    import torch
    print("torch:", torch.__version__)
except ImportError as e:
    print("torch: not installed:", e)
