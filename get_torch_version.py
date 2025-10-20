import torch
print("torch:", torch.__version__)
try:
    import whisper
    print("whisper:", whisper.__version__)
except Exception:
    print("whisper: not installed")
try:
    import pyannote.audio
    print("pyannote.audio:", pyannote.audio.__version__)
except Exception:
    print("pyannote.audio: not installed")
