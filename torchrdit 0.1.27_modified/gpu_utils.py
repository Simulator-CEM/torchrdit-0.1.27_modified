"""
GPU内存管理工具模块
提供CUDA预热和内存清理功能
"""

import gc
import torch
import time
from typing import Optional


def cuda_warmup(verbose: bool = True) -> None:
    """
    CUDA预热：触发JIT编译，避免首次运行计时不准
    
    Args:
        verbose: 是否打印详细信息
    """
    if not torch.cuda.is_available():
        if verbose:
            print("⚠️  CUDA不可用，跳过预热")
        return
    
    if verbose:
        print("🔥 正在预热CUDA...")
    
    device = torch.device('cuda')
    start = time.time()
    
    try:
        # 1. 预热cuFFT（FFT变换）
        dummy_fft = torch.randn(1000, 1000, device=device, dtype=torch.complex128)
        _ = torch.fft.fft2(dummy_fft)
        _ = torch.fft.ifft2(dummy_fft)
        
        # 2. 预热cuBLAS（矩阵乘法）
        dummy_matmul = torch.randn(500, 500, device=device, dtype=torch.complex128)
        _ = torch.matmul(dummy_matmul, dummy_matmul.conj().T)
        
        # 3. 预热cuSOLVER（线性求解器）
        dummy_solve_A = torch.randn(200, 200, device=device, dtype=torch.complex128)
        dummy_solve_B = torch.randn(200, 200, device=device, dtype=torch.complex128)
        _ = torch.linalg.solve(dummy_solve_A, dummy_solve_B)
        
        # 4. 预热Hessian计算（自动微分）
        def dummy_loss(x):
            return (x ** 2).sum()
        
        dummy_grad = torch.randn(100, device=device, requires_grad=True)
        loss = dummy_loss(dummy_grad)
        _ = torch.autograd.grad(loss, dummy_grad, create_graph=True)[0]
        
        # 5. 等待所有GPU操作完成
        torch.cuda.synchronize()
        
        # 6. 清理预热产生的缓存
        del dummy_fft, dummy_matmul, dummy_solve_A, dummy_solve_B, dummy_grad, loss
        torch.cuda.empty_cache()
        
        elapsed = time.time() - start
        
        if verbose:
            print(f"✅ CUDA预热完成！耗时 {elapsed:.2f} 秒")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   已分配内存: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            
    except Exception as e:
        if verbose:
            print(f"⚠️  CUDA预热失败: {e}")


def cleanup_gpu_memory(verbose: bool = True, aggressive: bool = False) -> None:
    """
    彻底清理GPU内存
    
    Args:
        verbose: 是否打印详细信息
        aggressive: 是否使用激进模式（会清理更多缓存，但可能影响后续性能）
    """
    if not torch.cuda.is_available():
        if verbose:
            print("⚠️  CUDA不可用，跳过清理")
        return
    
    if verbose:
        mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"🧹 正在清理GPU内存...")
        print(f"   清理前: {mem_before:.1f} MB")
    
    try:
        # 1. 强制Python垃圾回收
        gc.collect()
        
        # 2. 清空PyTorch的GPU缓存
        torch.cuda.empty_cache()
        
        # 3. 同步所有CUDA流
        torch.cuda.synchronize()
        
        # 4. 清理跨进程共享内存（IPC）
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
        
        # 5. 激进模式：重置峰值内存统计（可选）
        if aggressive:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        
        if verbose:
            mem_after = torch.cuda.memory_allocated() / 1024**2
            mem_cached = torch.cuda.memory_reserved() / 1024**2
            print(f"   清理后: {mem_after:.1f} MB")
            print(f"   缓存池: {mem_cached:.1f} MB")
            print(f"✅ 内存清理完成！释放了 {mem_before - mem_after:.1f} MB")
            
    except Exception as e:
        if verbose:
            print(f"⚠️  内存清理失败: {e}")


def get_gpu_memory_info() -> dict:
    """
    获取当前GPU内存使用信息
    
    Returns:
        包含内存信息的字典
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
    }


def print_gpu_memory_info() -> None:
    """打印GPU内存使用信息"""
    info = get_gpu_memory_info()
    
    if not info["available"]:
        print("⚠️  CUDA不可用")
        return
    
    print(f"\n📊 GPU内存使用情况:")
    print(f"   设备: {info['device_name']}")
    print(f"   总内存: {info['total_mb']:.0f} MB")
    print(f"   已分配: {info['allocated_mb']:.1f} MB ({info['allocated_mb']/info['total_mb']*100:.1f}%)")
    print(f"   已缓存: {info['reserved_mb']:.1f} MB ({info['reserved_mb']/info['total_mb']*100:.1f}%)")
    print(f"   峰值使用: {info['max_allocated_mb']:.1f} MB")


class GPUMemoryManager:
    """
    GPU内存管理上下文管理器
    使用方法:
        with GPUMemoryManager():
            # 你的GPU代码
            ...
    """
    
    def __init__(self, warmup: bool = True, cleanup: bool = True, verbose: bool = True):
        self.warmup = warmup
        self.cleanup = cleanup
        self.verbose = verbose
    
    def __enter__(self):
        if self.warmup:
            cuda_warmup(verbose=self.verbose)
        
        if self.verbose:
            print_gpu_memory_info()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup:
            cleanup_gpu_memory(verbose=self.verbose)
        
        if self.verbose:
            print_gpu_memory_info()


# 使用示例
if __name__ == "__main__":
    print("=" * 60)
    print("GPU工具模块测试")
    print("=" * 60)
    
    # 测试1: CUDA预热
    cuda_warmup()
    
    # 测试2: 创建一些GPU张量
    if torch.cuda.is_available():
        print("\n创建测试张量...")
        x = torch.randn(5000, 5000, device='cuda')
        y = torch.matmul(x, x.T)
        print_gpu_memory_info()
        
        # 测试3: 清理内存
        del x, y
        cleanup_gpu_memory()
    
    # 测试4: 使用上下文管理器
    print("\n" + "=" * 60)
    print("测试上下文管理器:")
    print("=" * 60)
    
    with GPUMemoryManager():
        if torch.cuda.is_available():
            z = torch.randn(3000, 3000, device='cuda')
            _ = torch.fft.fft2(z)
