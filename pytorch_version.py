import torch

# PyTorch version
print(f"PyTorch version: {torch.__version__}")

# CUDA version that PyTorch was built with
print(f"CUDA version: {torch.version.cuda}")

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# If CUDA is available, get additional info
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")