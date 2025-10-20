try:
    import pyannote.audio
    print("pyannote.audio:", pyannote.audio.__version__)
except ImportError as e:
    print("pyannote.audio: not installed:", e)
