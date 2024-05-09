# Setup
**CAUTION: strictly use base size models if you have RAM <=16GB** 
- Create a virtual Env.
- Install the requirements `pip install -r requirements.txt`.
- Create a folder called video in the cloned directory(this is the folder where videos should exist to test the benchmarking).
- Create another folder named chunked in the same directory.(this is required if you run the manual chunking script that mn_chunking.py)
- Run the script "python script.py".


# Report

test-details:  
- **_model-size_**: medium  
- **_number of threads_**: 4  
- **_beam_size_**: 1  
- **_cpu_**: 13th Gen Intel i5-13600KF (20) @ 5.100GHz   
- **_video_length_**: '6 mins'  
- **_RAM_**: '32 GB'  

Whisper:   
**_Time-taken_**:  133.71 seconds  
**_CPU_USAGE_**:  19.8%  
**_MEMORY_SPIKE_**: 460.055 MB  


Faster-Whisper:  
**_Time-taken_**:  4.04 seconds  
**_CPU_USAGE_**:  18.6%  
**_MEMORY_SPIKE_**: 135.039 MB  

Whisper-X:  
**_Time-taken_**:  117.92 seconds  
**_CPU_USAGE_**:  19.9%  
**_MEMORY_SPIKE_**: 359.5 MB  

Faster-Whispher-with-Chunking-optimization:  
**_Time-taken_**:  143.83 seconds  
**_CPU_USAGE_**:  19.1%  
**_MEMORY_SPIKE_**: 170 MB  


**_MEMORY_USAGE_**(While running the script): 41.7% - 43%





