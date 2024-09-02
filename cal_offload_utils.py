import torch
import time

# 确保使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建给定大小的 Tensor
tensor_size = (1024, 64, 128)  # 修改为你需要的大小
tensor_gpu = torch.randn(tensor_size, device=device)

# Profile 计算 offload 时间
start_time = time.time()
tensor_cpu = tensor_gpu.to('cpu')
offload_time = time.time() - start_time
print(f'Offload time (GPU -> CPU): {offload_time:.6f} seconds')

# 再次将 Tensor 载入 GPU
tensor_gpu = tensor_cpu.to(device)

# Profile 计算同时 offload 和 reload 的时间
start_time = time.time()

# 异步进行 offload 和 reload
tensor_cpu = tensor_gpu.to('cpu', non_blocking=True)
tensor_gpu_reload = tensor_cpu.to(device, non_blocking=True)

# 等待完成同步
torch.cuda.synchronize()

duplex_time = time.time() - start_time
print(f'Duplex communication time (offload + reload): {duplex_time:.6f} seconds')