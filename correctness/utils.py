import re
import json
import os
import time
import torch
import threading
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple, Callable

def normalize_text(s: str) -> str:
    """Normalize text by trimming and normalizing whitespace."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    return " ".join(s.split())

def simple_chunk(text: str, max_tokens: int = 250) -> List[str]:
    """Very simple English sentence splitting + length-based chunking 
    (approximately counting by spaces)"""
    if not text:
        return []
    parts, buf = [], []
    approx_len = 0
    for seg in text.replace("\n", " ").split(". "):
        seg = seg.strip()
        if not seg:
            continue
        seg_len = len(seg.split())
        if approx_len + seg_len > max_tokens and buf:
            parts.append(" ".join(buf).strip())
            buf = [seg]
            approx_len = seg_len
        else:
            buf.append(seg)
            approx_len += seg_len
    if buf:
        parts.append(" ".join(buf).strip())
    return parts

def set_torch_deterministic(seed: int = 42):
    """Set torch to be deterministic for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information in MB."""
    if not torch.cuda.is_available():
        return {"total": 0, "allocated": 0, "free": 0, "cached": 0}
    
    try:
        # Get memory info for current device
        device = torch.cuda.current_device()
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)  # MB
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)  # MB
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)  # MB
        cached = reserved - allocated
        free = total_mem - reserved
        
        return {
            "total": round(total_mem, 2),
            "allocated": round(allocated, 2),
            "free": round(free, 2),
            "cached": round(cached, 2)
        }
    except Exception as e:
        print(f"Error getting GPU memory info: {e}")
        return {"error": str(e)}

def pick_device(prefer_gpu: bool = True) -> str:
    """Select appropriate torch device based on preferences and availability."""
    if prefer_gpu and torch.cuda.is_available():
        # Print CUDA device information
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s):")
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            print(f"  - Device {i}: {device_properties.name} with {device_properties.total_memory / 1024**3:.1f} GB memory")
        return "cuda"
    if torch.backends.mps.is_available():
        print("Using Apple MPS (Metal Performance Shaders) device")
        return "mps"
    print("No GPU available, using CPU")
    return "cpu"

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def ensure_json_object(text: str) -> Dict:
    """
    Attempt to extract the outermost JSON object from model output, 
    preventing occasional extraneous prefixes/suffixes.
    """
    if not isinstance(text, str):
        return {}
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def save_metadata_jsonl(path: str, meta: List[Dict]):
    """Save metadata to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def load_metadata_jsonl(path: str) -> List[Dict]:
    """Load metadata from a JSONL file."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def run_with_timer(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Run a function with timer and return the result and elapsed time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time
    return result, elapsed

def run_parallel(func: Callable, items: List[Any], max_workers: int = None) -> List[Any]:
    """Run a function in parallel over multiple items.
    
    Args:
        func: The function to call for each item
        items: List of items to process
        max_workers: Maximum number of parallel workers (default: CPU count)
        
    Returns:
        List of results, one per input item
    """
    if max_workers is None:
        max_workers = min(os.cpu_count(), 4)  # Limit to avoid overloading
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(func, item): item for item in items}
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing {item}: {e}")
                results.append(None)
    return results

def parse_filename(filename: str) -> Dict[str, str]:
    """Parse filename into components based on '^' separator.
    
    Example: 'Apple^iphone^15pro^survey^questionnaire.json' ->  
    {'brand': 'Apple', 'product': 'iphone', 'model': '15pro', 'type': 'survey'}
    """
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    parts = basename.split('^')
    
    result = {
        "brand": parts[0] if len(parts) > 0 else "",
        "product": parts[1] if len(parts) > 1 else "",
        "model": parts[2] if len(parts) > 2 else "",
        "type": parts[3] if len(parts) > 3 else ""
    }
    return result

def get_brand_product_key(filename: str) -> str:
    """Extract brand^product key from filename."""
    info = parse_filename(filename)
    return f"{info['brand']}^{info['product']}"
