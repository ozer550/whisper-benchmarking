import subprocess
import whisper
from faster_whisper import WhisperModel
import time
from moviepy.editor import VideoFileClip
from colorama import init, Fore
import os
init()

whisper_model = whisper.load_model("base", device="cpu")
faster_whisper_model = WhisperModel("base", device="cpu", compute_type="float32")


def chunk_video(input_video_path, output_directory, chunk_duration):
    output_template = os.path.join(output_directory, f"chunk_%03d.mp4")
    command = [
        "ffmpeg",
        "-i", input_video_path,
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-reset_timestamps", "1",
        output_template
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("FFmpeg output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error chunking video {input_video_path}: {e}")
        print("FFmpeg error output:", e.stderr)
        raise e

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
    segments, info = model.transcribe(audio_path, beam_size=1)
    elapsed_time = time.time() - start_time
    texts = [segment.text for segment in segments]
    full_text = " ".join(texts)
    return full_text, info, elapsed_time


results = []


directory_non_chunked = './video'
directory_chunked_manually = './chunked'
chunked_directory = './chunked'
chunk_duration = 15
video_paths_non_chunked = [os.path.join(directory_non_chunked, file) for file in os.listdir(directory_non_chunked) if file.endswith(('.mp4', '.MP4'))]



def chunk_current_video(video_path_non_chunked):
    print(f"Chunking {video_path_non_chunked}...")
    os.makedirs(chunked_directory, exist_ok=True)
    chunk_video(video_path_non_chunked, chunked_directory, chunk_duration)
    video_paths_manually_chunked = [os.path.join(directory_chunked_manually, file) for file in os.listdir(directory_chunked_manually) if file.endswith(('.mp4', '.MP4'))]
    return video_paths_manually_chunked

# Process each video file
for video_path_non_chunked in video_paths_non_chunked: 
    
    video_paths_manually_chunked = chunk_current_video(video_path_non_chunked)
    video_filename = os.path.basename(video_path_non_chunked)
    print(f"\nProcessing {video_path_non_chunked}...")
    audio_path_non_chunked, video_duration_path_non_chunked = extract_audio(video_path_non_chunked)
    
    audio_paths_manually_chunked = [] 
    for chunk in video_paths_manually_chunked:
        print(f"\nProcessing {chunk}...")
        audio_path_manually_chunked, video_duration_path_manually_chunked = extract_audio(chunk)
        audio_paths_manually_chunked.append(audio_path_manually_chunked)
    
    # Transcribe with Whisper
    print("*******************************")
    print("Transcribing with Whisper...")
    print()
    whisper_result, whisper_time_non_chunked = transcribe_whisper(whisper_model, audio_path_non_chunked)
    print("non-chunked Transcribe function output:", whisper_result)
    print(Fore.YELLOW + f"Whisper transcription time: {whisper_time_non_chunked:.2f} seconds" + Fore.RESET)
    print()
    for audio in audio_paths_manually_chunked:
        whisper_result, whisper_time_manually_chunked = transcribe_whisper(whisper_model, audio)
        print("manually chunked Transcribe function output:", whisper_result)
        print()
        print(Fore.YELLOW + f"Whisper transcription time: {whisper_time_manually_chunked:.2f} seconds" + Fore.RESET)
    print()
    print()

    # Transcribe with Faster Whisper
    print("--------------------------------")
    print("Transcribing with Faster Whisper...")
    print()
    faster_whisper_result, faster_whisper_info, faster_whisper_time_non_chunked = transcribe_faster_whisper(faster_whisper_model, audio_path_non_chunked)
    print("Non Chunked Faster Whisper transcription output:", faster_whisper_result)
    print(Fore.GREEN + f"Faster Whisper transcription time: {faster_whisper_time_non_chunked:.2f} seconds" + Fore.RESET)
    print()
    for audio in audio_paths_manually_chunked:
        faster_whisper_result, faster_whisper_info, faster_whisper_time_manually_chunked = transcribe_faster_whisper(faster_whisper_model, audio)
        print("Manually Chunked Faster Whisper transcription output:", faster_whisper_result)
        print()
        print(Fore.GREEN + f"Faster Whisper transcription time: {faster_whisper_time_manually_chunked:.2f} seconds" + Fore.RESET)
    print()
    print()
    
    video_file_name = os.path.basename(video_path_non_chunked)
    video_duration_fmt = f"{video_duration_path_non_chunked:.2f} sec"
    
    # Remove all chunked files
    for video_path in video_paths_manually_chunked:
        os.remove(video_path)
