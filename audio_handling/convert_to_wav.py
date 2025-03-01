import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.ERROR)

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import numpy as np
import os
import soundfile as sf
from audio_handling.delete_chunks import delete_chunks  # Import the cleanup function

def split_and_save_audio(audio_path, chunk_length=20):
    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    samples_per_chunk = chunk_length * sample_rate
    total_samples = len(waveform)
    base_name = os.path.splitext(audio_path)[0]
    
    chunks = []
    for i in range(0, total_samples, samples_per_chunk):
        chunk = waveform[i:i + samples_per_chunk]
        if len(chunk) < samples_per_chunk:
            chunk = np.pad(chunk, (0, samples_per_chunk - len(chunk)), mode='constant')
        chunks.append(chunk)
    
    chunk_files = []
    for idx, chunk in enumerate(chunks):
        chunk_filename = f"{base_name}_{idx + 1:02d}.wav"
        sf.write(chunk_filename, chunk, sample_rate, subtype='PCM_16')
        chunk_files.append(chunk_filename)
    
    return chunk_files, sample_rate

def transcribe_chunks(chunk_files, model, processor, sample_rate):
    transcriptions = []
    for chunk_file in chunk_files:
        waveform, sr = librosa.load(chunk_file, sr=None, mono=True)
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000
        waveform = torch.tensor(waveform).unsqueeze(0)
        input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0].lower()
        #print(f"Transcription for {chunk_file}: {transcription}")
        transcriptions.append(transcription+"\n")
    return transcriptions

def transcribe_long_audio(audio_path, chunk_length=20, output_file="Transcripts/transcription.txt"):
    model_name = "facebook/wav2vec2-large-960h-lv60-self"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    chunk_files, sample_rate = split_and_save_audio(audio_path, chunk_length)
    transcriptions = transcribe_chunks(chunk_files, model, processor, sample_rate)
    
    full_transcription = " ".join(transcriptions)
   # print("\nFull Transcription:", full_transcription)
    
    with open(output_file, "w") as f:
        f.write(full_transcription)
    #print(f"Transcription saved to {output_file}")
    
    # Clean up chunk files
    delete_chunks(chunk_files)
    
    return full_transcription


def main_transcribe(audio_path):
    #audio_path = "/Users/denzhedzebu/Desktop/discover ai/audio_handling/Tana Road 2.m4a"
    a = transcribe_long_audio(audio_path)
    os.system("clear")
    return a