
import concurrent.futures
import time
import random
import json
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

# --- Mock ask_llm function ---
def ask_llm_gt(prompt, assumed_role="You are a helpful AI assistant."):
    """
    Mock function to simulate calling an LLM (GPT/Gemini).
    Replace this with your actual API call.
    """
    # Simulate network latency - random sleep between 0.1 and 1.0 seconds
    # Reduced from user's manual edit (2-14s) for testing purposes, 
    # but in real usage this would be actual API time.
    sleep_secs = random.uniform(2, 14)
    time.sleep(sleep_secs) 
    
    # Simple mock logic based on prompt content
    if "Can the following article answer the question?" in prompt:
        return random.choice(["Yes", "No"])
    
    return "I don't know"

# --- Core Logic ---

def check_relevance(question, article_text):
    """
    Determines if an article is relevant to the question using the LLM.
    """
    prompt = f"""
    You are an evaluator for a retrieval system.
    
    Question: {question}
    
    Article Content:
    {article_text[:2000]} # Truncate to avoid context limit issues in this mock
    
    Task:
    Can the following article answer the question?
    Reponse with ONLY "Yes" or "No".
    """
    
    try:
        response = ask_llm_gt(prompt, assumed_role="You are a strict evaluator.")
        clean_response = response.strip().lower()
        return "yes" in clean_response
    except Exception as e:
        # In case of LLM failure, we can return False or re-raise
        # For this script, we'll return False and log if needed
        return False

def process_single_pair(task_tuple):
    """
    Process a single (question, article_id, article_text) tuple.
    Returns: (question_text, article_id, is_relevant)
    """
    question_text, article_id, article_text = task_tuple
    is_relevant = check_relevance(question_text, article_text)
    return (question_text, article_id, is_relevant)

def generate_ground_truth(dataset, article_database, output_file: Path, batch_size=50, max_workers=20):
    """
    Generates ground truth using exploded parallelism and batching.
    Output is written to a JSONL file incrementally.
    Supports resumption by skipping already processed tasks.
    """
    
    # 1. Identify already processed tasks
    processed_keys = set()
    if output_file.exists():
        print(f"Reading existing progress from {output_file}...")
        try:
            with output_file.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        # Create a unique key for the task
                        key = (record['question'], record['article_id'])
                        processed_keys.add(key)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}")
            
    print(f"Found {len(processed_keys)} already processed tasks.")

    # 2. Explode the dataset into individual tasks and filter
    # Each task is: (question_text, article_id, article_text)
    all_tasks = []
    skipped_count = 0
    
    for q_idx, q_data in enumerate(dataset):
        question_text = q_data['question']
        for article_id in q_data['list_of_articleId']:
            # Skip if already processed
            if (question_text, article_id) in processed_keys:
                skipped_count += 1
                continue
                
            if article_id in article_database:
                all_tasks.append((question_text, article_id, article_database[article_id]))
            
    total_tasks = len(all_tasks)
    print(f"Total individual checks to perform: {total_tasks} (Skipped {skipped_count})")
    
    if total_tasks == 0:
        print("All tasks completed! Nothing to run.")
        return

    # 3. Setup Output File (Append mode)
    # The file is opened in append mode inside the loop/init
    
    # 4. Process in batches
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        expand=True
    ) as progress:
        
        overall_task = progress.add_task("[green]Processing Ground Truth...", total=total_tasks)
        
        # Iterate over tasks in chunks of batch_size
        for i in range(0, total_tasks, batch_size):
            batch = all_tasks[i : i + batch_size]
            
            batch_results = []
            
            # Parallel Execution for the batch
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Map returns results in order
                results_iter = executor.map(process_single_pair, batch)
                
                for res in results_iter:
                    batch_results.append(res)
                    progress.advance(overall_task)
            
            # Write batch results to file
            # Format: {"question": "...", "article_id": "...", "is_relevant": true/false}
            with output_file.open("a", encoding="utf-8") as f:
                for q_text, art_id, is_rel in batch_results:
                    record = {
                        "question": q_text,
                        "article_id": art_id,
                        "is_relevant": is_rel
                    }
                    f.write(json.dumps(record) + "\n")

    print(f"Completed! Results saved to {output_file}")

def consolidate_results(dataset, output_file: Path, final_output_file: Path):
    """
    Reads the incremental JSONL output, aggregates relevant articles,
    and updates the original dataset with a 'gt_docs' field.
    """
    print(f"\nConsolidating results from {output_file}...")
    
    # 1. Aggregate results from JSONL
    # Map: question_text -> set of relevant article_ids
    relevance_map = {}
    
    if not output_file.exists():
        print("Output file not found. Cannot consolidate.")
        return

    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                q_text = record['question']
                art_id = record['article_id']
                is_rel = record['is_relevant']
                
                if is_rel:
                    if q_text not in relevance_map:
                        relevance_map[q_text] = []
                    relevance_map[q_text].append(art_id)
            except json.JSONDecodeError:
                continue
                
    # 2. Update original dataset
    updated_count = 0
    for item in dataset:
        q_text = item['question']
        # Default to empty list if no relevant docs found or question not processed
        item['gt_docs'] = relevance_map.get(q_text, [])
        updated_count += 1
        
    print(f"Updated {updated_count} records with ground truth.")

    # 3. Save final dataset
    with final_output_file.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Final dataset saved to {final_output_file}")


if __name__ == "__main__":
    # --- Mock Data Generation ---
    random.seed(42)
    print("Generating mock data...")
    
    article_db = {}
    for i in range(100):
        article_db[f"msg_{i}"] = f"Article {i} content. Topic {i%5}."

    mock_dataset = []
    # Create enough data to see the progress bar move
    for i in range(50): 
        q_obj = {
            "question": f"Question {i} about topic {i%5}?",
            "list_of_articleId": [f"msg_{random.randint(0, 99)}" for _ in range(20)]
        }
        mock_dataset.append(q_obj)

    # Use Pathlib
    output_path = Path("ground_truth_output.jsonl")
    final_dataset_path = Path("final_dataset_with_gt.json")
    


    # --- Execution ---
    start_time = time.time()
    
    # Run the generator
    generate_ground_truth(mock_dataset, article_db, output_path, batch_size=10, max_workers=20)
    
    # Consolidate
    consolidate_results(mock_dataset, output_path, final_dataset_path)
    
    end_time = time.time()
    print(f"\nTime taken: {end_time - start_time:.2f} seconds")
    
    # Verify by reading a few lines
    print("\nVerifying final dataset (first record):")
    if final_dataset_path.exists():
        with final_dataset_path.open("r") as f:
            data = json.load(f)
            if data:
                print(json.dumps(data[0], indent=2))