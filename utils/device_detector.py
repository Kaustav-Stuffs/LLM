import torch
import logging
import time

logger = logging.getLogger(__name__)

def get_optimal_device(retry_attempts: int = 3) -> str:
    """
    Automatically detect and return the best available device for computation.
    
    Args:
        retry_attempts: Number of times to retry CUDA detection
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    for attempt in range(retry_attempts):
        try:
            # Clear CUDA cache if previous attempt failed
            if attempt > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(1)  # Short delay between attempts
            
            if torch.cuda.is_available():
                # Additional CUDA validation
                try:
                    torch.cuda.current_device()
                    torch.cuda.synchronize()
                    device = "cuda"
                    gpu_count = torch.cuda.device_count()
                    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                    logger.info(f"CUDA available: {gpu_count} GPU(s) - {gpu_name}")
                    return device
                except Exception as cuda_error:
                    logger.warning(f"CUDA validation failed on attempt {attempt + 1}: {cuda_error}")
                    if attempt == retry_attempts - 1:
                        logger.warning("CUDA available but validation failed, falling back to CPU")
                    continue
                    
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info("MPS (Apple Silicon) available")
                return device
            else:
                device = "cpu"
                logger.info("Using CPU")
                return device
                
        except Exception as e:
            logger.warning(f"Error detecting device on attempt {attempt + 1}: {e}")
            if attempt == retry_attempts - 1:
                logger.warning("All device detection attempts failed. Falling back to CPU.")
                return "cpu"
    
    return "cpu"
