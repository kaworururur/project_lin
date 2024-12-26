import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


wav = tts.tts(text="Я люблю читать книги", speaker_wav="my/cloning/", language="ru")
tts.tts_to_file(text="Я люблю читать книги", speaker_wav="my/cloning/audio3.wav", language="ru", file_path="fake4.wav")