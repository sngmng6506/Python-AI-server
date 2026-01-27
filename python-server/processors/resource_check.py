import psutil
import torch
import pynvml  # ✅ 올바른 import 방식
import logging

logger = logging.getLogger(__name__)

# ----------------------------
# NVML 초기화 
# ----------------------------

_nvml_initialized = False


def _ensure_nvml():
    """NVML 초기화 (에러 핸들링 포함)"""
    global _nvml_initialized
    
    if not _nvml_initialized:
        try:
            pynvml.nvmlInit()
            _nvml_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            return False
    
    return True


def check_resource_usage() -> dict:
    """
    Returns resource usage based on execution device.
    CPU  -> RAM %, CPU %
    GPU  -> VRAM %, GPU %
    """
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        return {
            "device": "gpu",
            "memory_percent": gpu_memory_occupancy(device_index),
            "compute_percent": gpu_compute_usage(device_index),
        }
    else:
        return {
            "device": "cpu",
            "memory_percent": ram_memory_occupancy(),
            "compute_percent": cpu_compute_usage(),
        }


# ----------------------------
# 메모리 점유율 (Admission control)
# ----------------------------


def ram_memory_occupancy() -> float:
    return round(psutil.virtual_memory().percent, 2)


def gpu_memory_occupancy(device_index: int) -> float:
    if not _ensure_nvml():
        return 0.0
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

        return round((mem.used  / mem.total)*100, 2)
        # return utilization.gpu
    except Exception as e:
        logger.error(f"Failed to get GPU memory occupancy: {e}")
        return 0.0


# ----------------------------
# 연산 사용률 (Monitoring)
# ----------------------------

def cpu_compute_usage() -> float:
    return round(psutil.cpu_percent(interval=0.1), 2)


def gpu_compute_usage(device_index: int) -> float:
    if not _ensure_nvml():
        return 0.0
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception as e:
        logger.error(f"Failed to get GPU compute usage: {e}")
        return 0.0
