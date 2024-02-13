import threading
from threading import Event, Thread
from transformers import pipeline, WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor
import torch
import record_audio
import time
import signal

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "CWKSC/whisper-tiny-common_voice_13_0-ja"
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")

transcriber = pipeline(
    "automatic-speech-recognition", model=model_id, tokenizer=tokenizer, device=device
)

loop = Event()
have_audio_to_transcribe = Event()

def transcribe():
    while not loop.is_set():
        have_audio_to_transcribe.wait()
        output = transcriber("recording.wav")
        print("text: ", output['text'])
        have_audio_to_transcribe.clear()
    print("transcribe thread end")

def recording():
    while not loop.is_set():
        if record_audio.record_system_sound(4, "recording.wav", loop):
            have_audio_to_transcribe.set()
    print("record thread end")

transcribe_thread = Thread(target=transcribe)
transcribe_thread.start()

recording_thread = Thread(target=recording)
recording_thread.start()

def handler(signum, frame):
    loop.set()
    exit()

signal.signal(signal.SIGINT, handler)
time.sleep(999999)

