try:
    import torchaudio
    print("torchaudio:", torchaudio.__version__)
    print("set_audio_backend:", hasattr(torchaudio, "set_audio_backend"))
except ImportError as e:
    print("torchaudio: not installed:", e)
