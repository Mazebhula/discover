import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress Wav2Vec2 warning
import logging
logging.getLogger("urllib3").setLevel(logging.ERROR)  # Suppress urllib3 warning

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

def transcribe(audio_path):
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
    waveform = torch.tensor(waveform).unsqueeze(0)
    input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print("Transcription:", transcription.lower())
    return transcription.lower()
if __name__ == "__main__":
    transcribe("")