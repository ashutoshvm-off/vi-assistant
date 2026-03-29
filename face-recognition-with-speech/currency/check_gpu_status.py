#!/usr/bin/env python3
"""
GPU Status Checker
Shows GPU and CPU utilization during training
"""
import tensorflow as tf
import psutil
import subprocess
import time
import threading

print("=" * 70)
print("GPU & CPU STATUS CHECKER")
print("=" * 70)

# Check GPU
print("\n[GPU INFORMATION]")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPUs Available: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
else:
    print("✗ No GPUs detected")

# Check CUDA
print("\n[CUDA INFORMATION]")
print(f"CUDA Available: {tf.test.is_built_with_cuda()}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# CPU Info
print("\n[CPU INFORMATION]")
cpu_percent = psutil.cpu_percent(interval=1)
print(f"CPU Usage: {cpu_percent}%")
print(f"CPU Count: {psutil.cpu_count()}")

# Memory Info
print("\n[MEMORY INFORMATION]")
mem = psutil.virtual_memory()
print(f"Total Memory: {mem.total / (1024**3):.2f} GB")
print(f"Available Memory: {mem.available / (1024**3):.2f} GB")
print(f"Memory Usage: {mem.percent}%")

# Try to get GPU memory if nvidia-smi available
print("\n[NVIDIA GPU MEMORY]")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            total, used, free = map(int, line.split(', '))
            print(f"GPU {i}:")
            print(f"  Total: {total / 1024:.2f} GB")
            print(f"  Used: {used / 1024:.2f} GB ({(used/total)*100:.1f}%)")
            print(f"  Free: {free / 1024:.2f} GB")
    else:
        print("nvidia-smi not found or failed to execute")
except Exception as e:
    print(f"Could not get nvidia-smi info: {e}")

# Test GPU computation
print("\n[GPU COMPUTATION TEST]")
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        c = tf.matmul(a, b)
    print("✓ GPU computation test PASSED")
    print(f"  Result: {c.numpy()}")
except Exception as e:
    print(f"✗ GPU computation test FAILED: {e}")

print("\n" + "=" * 70)
print("STATUS CHECK COMPLETE")
print("=" * 70)
print("\nFor training with GPU at ~5% and CPU at ~28% usage:")
print("  - Reduce batch size (default: 32)")
print("  - Reduce number of epochs")
print("  - Use mixed precision training")
print("  - Enable memory growth (already enabled)")
