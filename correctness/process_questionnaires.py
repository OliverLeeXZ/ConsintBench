import os
import re
import json
import time
import glob
import torch
import concurrent.futures
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
import shutil

from config import EmbedConfig, IndexPaths, OPENAI_KEYWORD_MODEL, EMBED_MODEL_NAME
from keyword_extraction import extract_keywords_with_gpt
from openai_api import get_answer
from build import run_build
from search import run_search
from utils import set_torch_deterministic, gpu_memory_info

def parse_questionnaire_name(filename: str) -> Dict[str, str]:
    """Parse questionnaire filename into components."""
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

def clean_discussion_data(input_file: str, output_file: str) -> bool:
    """Clean discussion data and save to output file."""
    print(f"[Clean] Processing {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        res = []
        for item in data:
            source = item['source']
            if source == 'twitter':
                dis = item['body'].get("full_text", "")
                if dis:
                    i = {
                        "source": source,
                        "discussion": dis,
                    }
                else:
                    continue
            else:
                dis = item['body'].get("content", "")
                if dis:
                    i = {
                        "source": source,
                        "discussion": dis,
                    }
                else:
                    continue
            res.append(i)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        
        print(f"[Clean] Successfully processed {input_file} -> {output_file}")
        return True
    except Exception as e:
        print(f"[Clean] Error processing {input_file}: {e}")
        return False

def clean_and_translate_comments(input_file: str) -> bool:
    """Additional cleaning and translation if needed."""
    # This is a placeholder for any additional cleaning steps
    # Based on the provided code snippet, it seems this might involve translation
    # or other transformations
    try:
        print(f"[Clean] Performing additional cleaning on {input_file}")
        # Implement additional cleaning logic here if needed
        return True
    except Exception as e:
        print(f"[Clean] Error in additional cleaning: {e}")
        return False

def get_embed_path(brand: str, product: str) -> str:
    """Get path to embedding directory for the given brand and product."""
    return os.path.join("embed", f"{brand}^{product}")

def get_discussion_path(brand: str, product: str) -> str:
    """Get path to raw discussion file for the given brand and product."""
    return os.path.join("discussion", f"{brand}^{product}.json")

def get_cleaned_discussion_path(brand: str, product: str) -> str:
    """Get path to cleaned discussion file for the given brand and product."""
    return os.path.join("cleaned_discussion", f"{brand}^{product}_cleaned.json")

def process_questionnaire(questionnaire_path: str) -> Dict:
    """Process a single questionnaire file."""
    start_time = time.time()
    
    # Parse questionnaire name
    q_info = parse_questionnaire_name(questionnaire_path)
    brand, product = q_info["brand"], q_info["product"]
    
    print(f"\n[Process] Starting {brand}^{product} questionnaire: {questionnaire_path}")
    
    # Check if embeddings exist
    embed_dir = get_embed_path(brand, product)
    cleaned_path = get_cleaned_discussion_path(brand, product)
    raw_path = get_discussion_path(brand, product)
    
    # Create artifacts directory within embed_dir
    artifacts_dir = os.path.join(embed_dir, "artifacts")
    
    # Step 1: Check if embeddings already exist
    if os.path.exists(embed_dir):
        print(f"[Process] Found existing embeddings at {embed_dir}")
    
    # Step 2: If no embeddings, check for cleaned discussions
    elif os.path.exists(cleaned_path):
        print(f"[Process] Found cleaned discussions at {cleaned_path}")
        # Create embedding directory
        os.makedirs(embed_dir, exist_ok=True)
        
        # Generate embeddings
        print(f"[Process] Generating embeddings from {cleaned_path}")
        cfg = EmbedConfig(
            model_name=EMBED_MODEL_NAME,
            batch_size=64,
            fp16=True,
            prefer_gpu=True,
            use_instruction=True,
            max_chunk_tokens=250
        )
        
        run_build(
            disc_path=cleaned_path,
            out_dir=artifacts_dir,
            cfg=cfg
        )
    
    # Step 3: If no cleaned discussions, check for raw discussions
    elif os.path.exists(raw_path):
        print(f"[Process] Found raw discussions at {raw_path}")
        
        # Create directory for cleaned discussions
        os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
        
        # Clean discussions
        clean_success = clean_discussion_data(raw_path, cleaned_path)
        
        if clean_success:
            # Additional cleaning if needed
            clean_and_translate_comments(cleaned_path)
            
            # Create embedding directory
            os.makedirs(embed_dir, exist_ok=True)
            
            # Generate embeddings
            print(f"[Process] Generating embeddings from {cleaned_path}")
            cfg = EmbedConfig(
                model_name=EMBED_MODEL_NAME,
                batch_size=64,
                fp16=True,
                prefer_gpu=True,
                use_instruction=True,
                max_chunk_tokens=250
            )
            
            run_build(
                disc_path=cleaned_path,
                out_dir=artifacts_dir,
                cfg=cfg
            )
        else:
            print(f"[Process] Failed to clean discussions for {brand}^{product}")
            return {
                "brand": brand,
                "product": product,
                "status": "failed",
                "error": "Failed to clean discussions"
            }
    
    else:
        print(f"[Process] No discussions found for {brand}^{product}")
        return {
            "brand": brand,
            "product": product,
            "status": "failed",
            "error": "No discussions found"
        }
    
    # Process questionnaire
    try:
        with open(questionnaire_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        
        array_data = [
            {
                "id": qid,
                "question": content.get("question", ""),
                "options": content.get("options", []),
                "answer": content.get("answer", "")
            }
            for qid, content in questions.items()
        ]
        
        N = min(25, len(array_data))
        answer_letters = []
        for i in range(N):
            ans = str(array_data[i]["answer"])
            answer_letters.append(ans.strip()[0] if ans else "?")
        
        # Generate compact keyword queries
        queries = [array_data[i]["question"] for i in range(N)]
        auto_queries = []
        for i, q in enumerate(queries):
            try:
                kw = extract_keywords_with_gpt(
                    question_text=q,
                    options=None,
                    model=OPENAI_KEYWORD_MODEL,  # Using config model for consistency
                    base_url=None,
                    max_items=8
                )
                ranked = (kw.get("keywords", []) or [])
                compact = " ".join(ranked).strip()
                
                # Fall back to original if extraction fails
                if not compact:
                    compact = q.strip()
                
                print(f"[Keywords] Q{i + 1} Original:", q)
                print(f"[Keywords] Q{i + 1} Ranked :", ranked)
                auto_queries.append(compact)
            
            except Exception as e:
                print(f"[Keywords] Q{i + 1} fallback due to error: {e}")
                auto_queries.append(q.strip())
        
        # Search configuration
        top_k = 10
        alpha = 0.20                       # engagement weight (0~0.3 common range)
        sources = ["twitter", "reddit"]    # source whitelist; empty = no filter
        
        # Use compact keywords for search
        result = run_search(
            out_dir=artifacts_dir,
            queries=auto_queries,
            top_k=top_k,
            alpha=alpha,
            source_whitelist=sources,
            retrieval_mode="hybrid",  # Changed to hybrid for better results
            keyword_weight=0.35,      # Balanced weight
            prefilter_k=200
        )
        
        # Get answers
        preds = []
        for i in range(N):
            q = str(array_data[i]["question"]) + " " + str(array_data[i]["options"])
            joined_text = "\n".join(result[i]) if i < len(result) else ""
            pred_line = get_answer(joined_text, q)
            print(i+1, pred_line)
            preds.append(pred_line)
        
        # Extract first letter for comparison
        pred_letters = [p[0] if isinstance(p, str) and len(p) > 0 else "?" for p in preds]
        
        # Calculate results
        wrong = 0
        wrong_questions = []
        for i in range(N):
            if pred_letters[i] != answer_letters[i]:
                wrong += 1
                wrong_questions.append({
                    "question_idx": i,
                    "question": array_data[i]["question"],
                    "options": array_data[i]["options"],
                    "gold": answer_letters[i],
                    "pred": pred_letters[i],
                    "raw_pred": preds[i]
                })
        
        accuracy = (N - wrong) / N if N > 0 else 0
        
        # Save results to file
        results_path = os.path.join(embed_dir, f"{brand}^{product}_results.json")
        results = {
            "brand": brand,
            "product": product,
            "total": N,
            "correct": N - wrong,
            "wrong": wrong,
            "accuracy": accuracy,
            "predictions": [{"question_id": i, "gold": answer_letters[i], "pred": pred_letters[i]} for i in range(N)],
            "wrong_questions": wrong_questions
        }
        
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[Summary] {brand}^{product} - Total: {N} | Correct: {N - wrong} | Wrong: {wrong} | Accuracy: {accuracy:.2%}")
        
        total_time = time.time() - start_time
        print(f"[Process] Completed {brand}^{product} in {total_time:.2f} seconds")
        
        return {
            "brand": brand,
            "product": product,
            "status": "success",
            "total": N,
            "correct": N - wrong,
            "wrong": wrong,
            "accuracy": accuracy,
            "time_taken": total_time
        }
        
    except Exception as e:
        print(f"[Process] Error processing questionnaire {questionnaire_path}: {e}")
        return {
            "brand": brand,
            "product": product,
            "status": "failed",
            "error": str(e)
        }

def main():
    """Main function to process all questionnaires in parallel."""
    # Setup directories
    os.makedirs("embed", exist_ok=True)
    os.makedirs("cleaned_discussion", exist_ok=True)
    
    # Find all questionnaire files
    questionnaire_files = glob.glob(os.path.join("questionnaire", "*questionnaire.json"))
    print(f"[Main] Found {len(questionnaire_files)} questionnaire files")
    
    if not questionnaire_files:
        print("[Main] No questionnaire files found in the 'questionnaire' directory")
        return
    
    # Process questionnaires in parallel
    max_workers = min(os.cpu_count(), 4)  # Limit to avoid overloading the system
    print(f"[Main] Starting processing with {max_workers} workers")
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_questionnaire = {
            executor.submit(process_questionnaire, qfile): qfile 
            for qfile in questionnaire_files
        }
        
        for future in concurrent.futures.as_completed(future_to_questionnaire):
            qfile = future_to_questionnaire[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                q_info = parse_questionnaire_name(qfile)
                results.append({
                    "brand": q_info["brand"],
                    "product": q_info["product"],
                    "status": "failed",
                    "error": str(e)
                })
                print(f"[Main] Error processing {qfile}: {e}")
    
    # Summarize results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print("\n===== Processing Summary =====")
    print(f"Total questionnaires: {len(results)}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_accuracy = sum(r["accuracy"] for r in successful) / len(successful)
        print(f"Average accuracy: {avg_accuracy:.2%}")
    
    if failed:
        print("\nFailed questionnaires:")
        for f in failed:
            print(f" - {f['brand']}^{f['product']}: {f.get('error', 'Unknown error')}")
    
    # Save overall results
    with open("processing_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n[Main] Processing completed. Results saved to processing_results.json")

if __name__ == "__main__":
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"[GPU] Available: {torch.cuda.device_count()} device(s)")
        print(f"[GPU] Memory: {gpu_memory_info()}")
    else:
        print("[GPU] Not available, using CPU (slower)")
    
    main()
