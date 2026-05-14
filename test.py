import torch

# 1. Check if CUDA (NVIDIA GPU support) is available
print(f"Is CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 2. Get GPU details
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # 3. Create a small tensor and move it to GPU to verify
    device = torch.device("cuda")
    x = torch.ones(3, 3).to(device)
    print("Successfully moved tensor to GPU:")
    print(x)
else:
    print("CUDA is not available. PyTorch is running on CPU.")