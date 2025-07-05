import torch
import psutil
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    info = {
        "python_version": "",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0,
        "cuda_devices": [],
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "memory_used_percent": psutil.virtual_memory().percent
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["current_device"] = torch.cuda.current_device()
        
        # Get info for each GPU
        for i in range(torch.cuda.device_count()):
            device_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_mb": torch.cuda.get_device_properties(i).total_memory // (1024**2),
                "memory_allocated_mb": torch.cuda.memory_allocated(i) // (1024**2),
                "memory_cached_mb": torch.cuda.memory_reserved(i) // (1024**2)
            }
            info["cuda_devices"].append(device_info)
    
    return info

def get_model_memory_usage(model) -> Dict[str, Any]:
    """Estimate model memory usage"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory (rough calculation)
        param_size_mb = total_params * 4 / (1024**2)  # Assuming float32
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_size_mb": round(param_size_mb, 2)
        }
    except Exception as e:
        return {"error": str(e)}

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")

def format_bytes(bytes_val: int) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"

def validate_generation_params(
    max_length: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    model_max_length: int = 2048
) -> Dict[str, Any]:
    """Validate and adjust generation parameters"""
    params = {}
    
    # Validate max_length
    if max_length is not None:
        params["max_length"] = min(max_length, model_max_length)
    else:
        params["max_length"] = min(100, model_max_length)
    
    # Validate temperature
    if temperature is not None:
        params["temperature"] = max(0.1, min(temperature, 2.0))
    else:
        params["temperature"] = 0.7
    
    # Validate top_p
    if top_p is not None:
        params["top_p"] = max(0.1, min(top_p, 1.0))
    else:
        params["top_p"] = 0.9
    
    # Validate top_k
    if top_k is not None:
        params["top_k"] = max(1, min(top_k, 100))
    else:
        params["top_k"] = 50
    
    return params