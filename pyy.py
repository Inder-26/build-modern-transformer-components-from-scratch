import torch
print(f'PyTorch Version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print('Apple Silicon MPS available')
else:
    print('Running on CPU') 