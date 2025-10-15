import os
import sys
import time
import json
import glob
import torch
import argparse
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Optional

from config import EmbedConfig, IndexPaths, ProcessingConfig, DirectoryPaths
from process_questionnaires import process_questionnaire
from utils import gpu_memory_info, get_brand_product_key, parse_filename

def setup_directories(dirs: DirectoryPaths):
    """Create necessary directories if they don't exist."""
    for dir_path in [
        dirs.questionnaire_dir,
        dirs.discussion_dir, 
        dirs.cleaned_dir,
        dirs.embed_dir,
        dirs.output_dir
    ]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"[Setup] Directory {dir_path} is ready")

def find_questionnaires(dirs: DirectoryPaths) -> List[str]:
    """Find all questionnaire files in the questionnaire directory."""
    pattern = os.path.join(dirs.questionnaire_dir, "*questionnaire.json")
    files = glob.glob(pattern)
    return sorted(files)

def process_all_questionnaires(
    questionnaire_files: List[str], 
    proc_cfg: ProcessingConfig,
    dirs: DirectoryPaths
) -> List[Dict]:
    """Process all questionnaire files in parallel."""
    if not questionnaire_files:
        print("[Main] No questionnaire files found")
        return []
    
    print(f"[Main] Found {len(questionnaire_files)} questionnaire files")
    for qf in questionnaire_files:
        print(f"  - {os.path.basename(qf)}")
    print()
    
    if not proc_cfg.enable_parallel or len(questionnaire_files) == 1:
        print("[Main] Running in sequential mode")
        results = []
        for qf in questionnaire_files:
            result = process_questionnaire(qf)
            results.append(result)
        return results
    
    # Process in parallel
    max_workers = min(len(questionnaire_files), proc_cfg.max_workers)
    print(f"[Main] Processing in parallel with {max_workers} workers")
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_questionnaire = {
            executor.submit(process_questionnaire, qf): qf 
            for qf in questionnaire_files
        }
        
        for future in concurrent.futures.as_completed(future_to_questionnaire):
            qfile = future_to_questionnaire[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[Main] Completed {os.path.basename(qfile)}")
            except Exception as e:
                q_info = parse_filename(qfile)
                error_result = {
                    "brand": q_info["brand"],
                    "product": q_info["product"],
                    "status": "failed",
                    "error": str(e)
                }
                results.append(error_result)
                print(f"[Main] Error processing {qfile}: {e}")
    
    return results

def save_results(results: List[Dict], dirs: DirectoryPaths):
    """Save processing results to output file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(dirs.output_dir, f"processing_results_{timestamp}.json")
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[Main] Results saved to {results_file}")

def summarize_results(results: List[Dict]):
    """Print summary of processing results."""
    if not results:
        print("[Main] No results to summarize")
        return
    
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]
    
    print("\n===== Processing Summary =====")
    print(f"Total questionnaires processed: {len(results)}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        total_accuracy = sum(r.get("accuracy", 0) for r in successful)
        avg_accuracy = total_accuracy / len(successful)
        print(f"Average accuracy: {avg_accuracy:.2%}")
        
        # Show top performing brands/products
        if len(successful) > 1:
            sorted_by_accuracy = sorted(successful, key=lambda r: r.get("accuracy", 0), reverse=True)
            print("\nTop performing questionnaires:")
            for i, r in enumerate(sorted_by_accuracy[:3], start=1):
                print(f"{i}. {r['brand']}^{r['product']}: {r.get('accuracy', 0):.2%} ({r.get('correct', 0)}/{r.get('total', 0)})")
    
    if failed:
        print("\nFailed questionnaires:")
        for f in failed:
            print(f" - {f['brand']}^{f['product']}: {f.get('error', 'Unknown error')}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run parallel processing of questionnaires")
    parser.add_argument("--sequential", action="store_true", help="Run in sequential mode instead of parallel")
    parser.add_argument("--workers", type=int, default=None, help="Maximum number of worker processes")
    parser.add_argument("--gpu-id", type=str, default=None, help="GPU ID(s) to use (comma-separated)")
    parser.add_argument("--questionnaire", type=str, help="Process a specific questionnaire file")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Print system info
    print("\n===== Survey Analysis System with GPU Acceleration =====")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configure GPU
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        print(f"[GPU] Using GPU(s): {args.gpu_id}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_names = [torch.cuda.get_device_properties(i).name for i in range(device_count)]
        print(f"[GPU] {device_count} GPU(s) detected: {', '.join(device_names)}")
        print(f"[GPU] Memory: {gpu_memory_info()}")
    else:
        print("[GPU] No GPU detected, running in CPU mode (significantly slower)")
    
    # Configuration
    dirs = DirectoryPaths()
    proc_cfg = ProcessingConfig()
    
    # Override config with arguments
    if args.sequential:
        proc_cfg.enable_parallel = False
    
    if args.workers:
        proc_cfg.max_workers = args.workers
    
    # Setup directories
    setup_directories(dirs)
    
    # Process questionnaires
    if args.questionnaire:
        # Process single questionnaire
        if not os.path.exists(args.questionnaire):
            print(f"[Error] Questionnaire file {args.questionnaire} does not exist")
            return
        
        print(f"[Main] Processing single questionnaire: {args.questionnaire}")
        results = [process_questionnaire(args.questionnaire)]
    else:
        # Find and process all questionnaires
        questionnaire_files = find_questionnaires(dirs)
        results = process_all_questionnaires(questionnaire_files, proc_cfg, dirs)
    
    # Save results
    save_results(results, dirs)
    
    # Summarize results
    summarize_results(results)
    
    print(f"\nProcessing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
