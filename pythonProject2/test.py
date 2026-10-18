import torch

print("Torch 版本:", torch.__version__)
print("CUDA 版本:", torch.version.cuda)
print("CUDA 是否可用:", torch.cuda.is_available())
print("设备名称:", torch.cuda.get_device_name(0))
print("设备计算能力:", torch.cuda.get_device_capability(0))  # 应显示 (12, 0)
print("支持的架构列表:", torch.cuda.get_arch_list())        # 必须包含 'sm_120'

# 简单的张量运算测试
if torch.cuda.is_available():
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = x @ y
    print("GPU 矩阵乘法测试通过，结果形状:", z.shape)