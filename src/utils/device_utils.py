import torch

def select_device(algo, policy="MlpPolicy", batch_size=None, history_len=1, net_arch=None, force_device="auto"):
    """
    Dynamically select the execution device (CPU vs GPU/CUDA) based on suitability.
    
    GPU is deemed suitable if:
      1. CUDA is available.
      2. And at least one of the following is true:
         - Recurrent policy or LSTM/sequence architecture is used (very suitable for GPU).
         - Batch size is large (e.g. batch_size >= 256) where GPU tensor parallelism benefits.
         - Stacked history length is large (history_len > 5) resulting in higher dimensional states.
         - The policy network architecture is large (e.g., hidden layers total units >= 512).
         
    Otherwise, CPU is selected because for small MLP policies (e.g. [64, 64]) in standard RL environments,
    CPU is faster due to the overhead of CPU-GPU data transfers across sequential environment steps.
    """
    if force_device is not None and force_device != "auto":
        # Respect explicit overrides
        if force_device == "cuda" and not torch.cuda.is_available():
            print("[Device Selection] CUDA was explicitly requested but is not available. Falling back to CPU.")
            return "cpu"
        return force_device
        
    if not torch.cuda.is_available():
        return "cpu"
        
    # Check if recurrence/sequence is used
    is_recurrent = "lstm" in policy.lower() or "recurrent" in policy.lower() or algo.lower() == "lstm"
    
    # Check network architecture size if specified
    is_large_arch = False
    if net_arch:
        if isinstance(net_arch, list):
            if sum(net_arch) >= 512:
                is_large_arch = True
        elif isinstance(net_arch, dict):
            pi_sum = sum(net_arch.get("pi", []))
            vf_sum = sum(net_arch.get("vf", []))
            if pi_sum >= 512 or vf_sum >= 512:
                is_large_arch = True
                
    # Determine suitability
    if is_recurrent:
        print(f"[Device Selection] Selecting GPU (CUDA) because recurrent/LSTM architecture is suitable.")
        return "cuda"
    elif history_len > 5:
        print(f"[Device Selection] Selecting GPU (CUDA) because history_len ({history_len}) is large.")
        return "cuda"
    elif batch_size is not None and batch_size >= 256:
        print(f"[Device Selection] Selecting GPU (CUDA) because batch_size ({batch_size}) is large.")
        return "cuda"
    elif is_large_arch:
        print(f"[Device Selection] Selecting GPU (CUDA) because policy network architecture is large.")
        return "cuda"
    else:
        # Default fallback for small MLP with small batch sizes
        print(f"[Device Selection] Selecting CPU because small MLP/batch size training is faster on CPU (avoiding GPU transfer overhead).")
        return "cpu"
