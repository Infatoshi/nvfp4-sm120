# Source before running Transformer Engine on Anvil.
# System CUDA 13.2 libs FIRST (newer cublasLt has cublasLtGroupedMatrixLayoutInit_internal
# that TE 2.15 needs); pip-packaged NVIDIA libs after for cudnn/nccl/etc.
export CUDA_HOME=/usr/local/cuda-13
export CUDNN_PATH=/home/infatoshi/.local/lib/python3.12/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH=/usr/local/cuda-13/targets/x86_64-linux/lib:/usr/local/cuda-13/lib64:/home/infatoshi/.local/lib/python3.12/site-packages/nvidia/cu13/lib:/home/infatoshi/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/infatoshi/.local/lib/python3.12/site-packages/nvidia/cusparselt/lib:/home/infatoshi/.local/lib/python3.12/site-packages/nvidia/nccl/lib:/home/infatoshi/.local/lib/python3.12/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=/usr/local/cuda-13/targets/x86_64-linux/lib/libcublasLt.so.13${LD_PRELOAD:+:$LD_PRELOAD}
