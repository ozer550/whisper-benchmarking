from faster_whisper import WhisperModel
import time
from moviepy.editor import VideoFileClip
from colorama import init, Fore
import os
from tabulate import tabulate
from parallelization import transcribe_audio
import psutil


init()

directory = './video'
video_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.mp4', '.MP4'))]

faster_whisper_model = WhisperModel("medium", device="cpu", compute_type="float32")


core_count_os = os.cpu_count()


def extract_audio(video_path):
    with VideoFileClip(video_path) as video:
        audio = video.audio
        audio_path = f"{video_path.rsplit('.', 1)[0]}_audio.wav"
        audio.write_audiofile(audio_path, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
        video_duration = video.duration
    return audio_path, video_duration


def transcribe_faster_whisper(model, audio_path):
    process = psutil.Process() 
    
    process.cpu_percent(None)
    memory_before = process.memory_info().rss / (1024 ** 2)
    
    start_time = time.time() 
    segments, info = model.transcribe(audio_path, beam_size=1) 
    elapsed_time = time.time() - start_time 
    
     
    cpu_used = process.cpu_percent(None)/core_count_os
    memory_after = process.memory_info().rss / (1024 ** 2)
    memory_used = memory_after - memory_before
    memory_percent_usage = psutil.virtual_memory().percent
     
    print()
    print("*******************************")
    print(tabulate([[cpu_used, memory_used, memory_percent_usage]], headers=["AVG CPU USAGE", "MEMORY SPIKE IN MB", "MEMORY USAGE"]))     
    print("*******************************")
    print()
    
    texts = [segment.text for segment in segments]
    full_text = " ".join(texts)
    return full_text, info, elapsed_time

def transcribe_faster_whisper_chunked(audio_path, model, max_processes=4):
    silence_threshold = "-20dB"
    silence_duration = 2  # seconds
    process = psutil.Process() 
    
    process.cpu_percent(None)
    memory_before = process.memory_info().rss / (1024 ** 2)
    
    start_time = time.time()
    full_text = transcribe_audio(audio_path, max_processes, silence_threshold, silence_duration, model)
    elapsed_time = time.time() - start_time
     
    cpu_used = process.cpu_percent(None)/core_count_os
    memory_after = process.memory_info().rss / (1024 ** 2)
    memory_used = memory_after - memory_before
    memory_percent_usage = psutil.virtual_memory().percent
     
    print()
    print("*******************************")
    print(tabulate([[cpu_used, memory_used, memory_percent_usage]], headers=["AVG CPU USAGE", "MEMORY SPIKE IN MB", "MEMORY USAGE"]))     
    print("*******************************")
    print() 
    
    return full_text, elapsed_time

results = []

# Process each video file
for video_path in video_paths:
    print(f"\nProcessing {video_path}...")
    audio_path, video_duration = extract_audio(video_path)

    # Transcribe with Faster Whisper
    print()
    print("--------------------------------")
    print("Transcribing with Faster Whisper...")
    print()
    faster_whisper_result, faster_whisper_info, faster_whisper_time = transcribe_faster_whisper(faster_whisper_model, audio_path)
    print("Faster Whisper transcription output:", faster_whisper_result)
    print()
    print(Fore.GREEN + f"Faster Whisper transcription time: {faster_whisper_time:.2f} seconds" + Fore.RESET)
    print()

    # Transcribe with Chunked Faster Whisper
    print()
    print("++++++++++++++++++++++++++++++++")
    print("Transcribing with Chunked Faster Whisper...")
    chunked_whisper_result, chunked_whisper_time = transcribe_faster_whisper_chunked(audio_path, faster_whisper_model)
    print("Chunked Faster Whisper transcription output:", chunked_whisper_result)
    print(Fore.CYAN + f"Chunked Faster Whisper transcription time: {chunked_whisper_time:.2f} seconds" + Fore.RESET)
    print()

    video_file_name = os.path.basename(video_path)
    video_duration_fmt = f"{video_duration:.2f} sec"
    results.append([video_file_name, video_duration_fmt, f"{faster_whisper_time:.2f} seconds", f"{chunked_whisper_time:.2f} seconds"])


print(tabulate(results, headers=["Video File", "Duration", "Faster Whisper Time", "Chunked Faster Whisper Time"]))
