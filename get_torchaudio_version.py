try:
    import torchaudio
    print("torchaudio:", torchaudio.__version__)
except ImportError as e:
    print("torchaudio: not installed:", e)
