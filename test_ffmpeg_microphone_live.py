from transformers.pipelines.audio_utils import ffmpeg_microphone_live

microphone = ffmpeg_microphone_live(sampling_rate=16000, chunk_length_s=5)

for chunk in microphone:
    print(chunk)