"""Fail closed unless a real RTX 5080 and native Blackwell CUDA work."""
import json
import torch


def check_gpu():
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("CUDA required; no CPU fallback permitted")
    if tuple(int(part) for part in torch.version.cuda.split(".")[:2]) < (12, 8):
        raise RuntimeError("CUDA 12.8+ required for Blackwell")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Nomad must allocate exactly one visible GPU")
    device = torch.cuda.get_device_properties(0)
    if "RTX 5080" not in device.name or (device.major, device.minor) != (12, 0):
        raise RuntimeError("RTX 5080 compute capability 12.0 required")
    if "sm_120" not in torch.cuda.get_arch_list():
        raise RuntimeError("PyTorch wheel lacks native sm_120 Blackwell kernels")
    a = torch.randn((256, 256), device="cuda")
    result = a @ a.T
    torch.cuda.synchronize()
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("CUDA compute preflight failed")
    print(json.dumps({"event": "gpu_preflight", "device": device.name,
                      "compute_capability": "12.0", "torch": torch.__version__,
                      "cuda": torch.version.cuda}), flush=True)


if __name__ == "__main__":
    check_gpu()
