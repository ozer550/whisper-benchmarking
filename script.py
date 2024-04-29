import whisper
from faster_whisper import WhisperModel
import time
from moviepy.editor import VideoFileClip
from colorama import init, Fore
import os
from tabulate import tabulate
import whisperx

init()

directory = './video'
video_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.mp4', '.MP4'))]

whisper_model = whisper.load_model("base", device="cpu")
faster_whisper_model = WhisperModel("base", device="cpu", compute_type="float32")
whisperx_model = whisperx.load_model("base", device="cpu", compute_type="float32")


def extract_audio(video_path):
    with VideoFileClip(video_path) as video:
        audio = video.audio
        audio_path = f"{video_path.rsplit('.', 1)[0]}_audio.wav"
        audio.write_audiofile(audio_path, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
        video_duration = video.duration
    return audio_path, video_duration

def transcribe_whisper(model, audio_path):
    start_time = time.time()
    result = model.transcribe(audio_path, beam_size=5)
    elapsed_time = time.time() - start_time
    if isinstance(result, dict) and 'text' in result:
        full_text = result['text']
    else:
        full_text = "No text extracted."
    return full_text, elapsed_time

def transcribe_faster_whisper(model, audio_path):
    start_time = time.time()
    segments, info = model.transcribe(audio_path, beam_size=5)
    elapsed_time = time.time() - start_time
    texts = [segment.text for segment in segments]
    full_text = " ".join(texts)
    return full_text, info, elapsed_time

def transcribe_whisperx(model, audio_path):
    start_time = time.time()
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=1)  
    elapsed_time = time.time() - start_time
    segments = result['segments']
    full_text = " ".join([seg['text'] for seg in segments])
    return full_text, elapsed_time


results = []

# Process each video file
for video_path in video_paths:
    print(f"\nProcessing {video_path}...")
    audio_path, video_duration = extract_audio(video_path)

    # Transcribe with Whisper
    print("*******************************")
    print("Transcribing with Whisper...")
    print()
    whisper_result, whisper_time = transcribe_whisper(whisper_model, audio_path)
    print("Transcribe function output:", whisper_result)
    print()
    print(Fore.YELLOW + f"Whisper transcription time: {whisper_time:.2f} seconds" + Fore.RESET)
    print()
    print()

    # Transcribe with Faster Whisper
    print("--------------------------------")
    print("Transcribing with Faster Whisper...")
    print()
    faster_whisper_result, faster_whisper_info, faster_whisper_time = transcribe_faster_whisper(faster_whisper_model, audio_path)
    print("Faster Whisper transcription output:", faster_whisper_result)
    print()
    print(Fore.GREEN + f"Faster Whisper transcription time: {faster_whisper_time:.2f} seconds" + Fore.RESET)
    print()
    print()
    
    # Transcribe with Whisper X
    print("================================")
    print("Transcribing with Whisper X...")
    whisperx_result, whisperx_time = transcribe_whisperx(whisperx_model, audio_path)
    print("Whisper X transcription output:", whisperx_result)
    print(Fore.BLUE + f"Whisper X transcription time: {whisperx_time:.2f} seconds" + Fore.RESET)
    
    video_file_name = os.path.basename(video_path)
    video_duration_fmt = f"{video_duration:.2f} sec"
    results.append([video_file_name, video_duration_fmt, f"{whisper_time:.2f} seconds", f"{faster_whisper_time:.2f} seconds", f"{whisperx_time:.2f} seconds"])

print(tabulate(results, headers=["Video File", "Duration", "Whisper Time", "Faster Whisper Time", "Whisper X Time"]))
