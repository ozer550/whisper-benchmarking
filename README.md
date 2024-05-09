# Setup
- Create a virtual Env.
- Install the requirements `pip install -r requirements.txt`.
- Create a folder called video in the cloned directory(this is the folder where videos should exist to test the benchmarking).
- Run the script "python script.py".


# Report

test-details:

- **_model-size_**: medium
- **_beam_size_**: 1
- **_cpu_**: 13th Gen Intel i7-13700HX (24) @ 4.800GHz 
- **_video_length_**: '309.34 secs'
- **_RAM_**: '16 GB'


Faster-Whisper:
**_Time-taken_**:  4.35 seconds
**_CPU_USAGE_**:  17.7%
**_MEMORY_SPIKE_**: 135.039 MB
**_MEMORY_USAGE_**: 72-74%

Faster-Whispher-with-Chunking-optimization:
**_Time-taken_**:  130.39 seconds
**_CPU_USAGE_**:  16.7%
**_MEMORY_SPIKE_**: 170 MB
**_MEMORY_USAGE_**: 72-74%






