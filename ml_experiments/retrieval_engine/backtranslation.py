import json
import random
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

class DataAugmenter:
    def __init__(self, t5_model_name="google-t5/t5-3b", sim_model_name="all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device found: {self.device}")

        # 1. Load Translation Model (Generator)
        print(f"Loading T5 model: {t5_model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, legacy=False)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.t5_model.eval()

        # 2. Load Similarity Model (Discriminator)
        print(f"Loading Similarity model: {sim_model_name}...")
        self.sim_model = SentenceTransformer(sim_model_name, device=self.device)

        # Supported Pivot Languages (T5 standard keys)
        # 'de': German, 'fr': French, 'ro': Romanian
        self.pivot_options = ["de", "fr", "ro"]
        self.lang_map = {"de": "German", "fr": "French", "ro": "Romanian", "en": "English"}

    def t5_generate(self, texts, target_languages):
        """
        Helper to run T5 generation for a batch of texts with varying target languages.
        """
        # Construct prompts: "translate English to German: {text}"
        input_texts = [
            f"translate English to {self.lang_map[lang]}: {text}" 
            if lang != "en" else 
            f"translate {self.lang_map[src_lang]} to English: {text}"
            for text, lang, src_lang in zip(texts, target_languages, ["en"]*len(texts) if target_languages[0] != "en" else texts) 
            # Note: The prompt logic is simplified for En->Pivot and Pivot->En below
        ]
        
        # Proper Prompt Construction
        prompts = []
        for text, tgt_lang in zip(texts, target_languages):
            if tgt_lang == "en": 
                # We don't know the source easily here without passing it, 
                # but T5 usually handles "translate German to English" well.
                # For simplicity in this specific loop, we handle prefixes in the main method.
                pass 
            else:
                prompts.append(f"translate English to {self.lang_map[tgt_lang]}: {text}")

        # Tokenize
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.t5_model.generate(
                **inputs, 
                max_length=512, 
                num_beams=2, # Reduced beams for speed
                temperature=0.9,
                do_sample=True # Add noise for diversity
            )
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def back_translate_batch(self, texts):
        """
        Performs En -> Random Pivot -> En loop.
        Returns: List of back-translated texts.
        """
        # 1. Assign Random Pivot for each text
        pivots = [random.choice(self.pivot_options) for _ in texts]
        
        # 2. Forward: English -> Pivot
        prompts_forward = [f"translate English to {self.lang_map[p]}: {t}" for p, t in zip(pivots, texts)]
        inputs_fwd = self.tokenizer(prompts_forward, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs_fwd = self.t5_model.generate(**inputs_fwd, max_length=512, do_sample=True, top_p=0.95)
        pivot_texts = self.tokenizer.batch_decode(outputs_fwd, skip_special_tokens=True)
        
        # 3. Backward: Pivot -> English
        prompts_back = [f"translate {self.lang_map[p]} to English: {t}" for p, t in zip(pivots, pivot_texts)]
        inputs_back = self.tokenizer(prompts_back, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs_back = self.t5_model.generate(**inputs_back, max_length=512, do_sample=True, top_p=0.95)
        
        return self.tokenizer.batch_decode(outputs_back, skip_special_tokens=True)

    def augment_dataset(self, json_data, batch_size=8):
        augmented_rows = []
        
        # Iterate through data in batches
        for i in tqdm(range(0, len(json_data), batch_size), desc="Augmenting"):
            batch_entries = json_data[i : i + batch_size]
            
            # Extract positives and negatives to process in bulk
            positives = [e['positive'] for e in batch_entries]
            negatives = [e['negative'] for e in batch_entries]
            
            # 1. Generate Candidates
            aug_positives = self.back_translate_batch(positives)
            aug_negatives = self.back_translate_batch(negatives)
            
            # 2. Compute Similarity Scores (Original vs Augmented)
            # Embed all at once for efficiency
            all_orig = positives + negatives
            all_aug = aug_positives + aug_negatives
            
            embeddings_orig = self.sim_model.encode(all_orig, convert_to_tensor=True)
            embeddings_aug = self.sim_model.encode(all_aug, convert_to_tensor=True)
            
            # Calculate Cosine Similarity per pair
            # diag gives us the similarity between row i of A and row i of B
            sim_scores = util.pairwise_cos_sim(embeddings_orig, embeddings_aug)
            
            # Split scores back into positive group and negative group
            pos_scores = sim_scores[:len(positives)]
            neg_scores = sim_scores[len(positives):]

            # 3. Filter and Construct New Rows
            for idx, entry in enumerate(batch_entries):
                query = entry['query']
                
                # Check Positive Augmentation
                p_score = pos_scores[idx].item()
                if 0.8 < p_score < 0.95:
                    augmented_rows.append({
                        "query": query,
                        "positive": aug_positives[idx], # New Positive
                        "negative": entry['negative'],  # Old Negative
                        "meta": "aug_pos"
                    })
                
                # Check Negative Augmentation
                n_score = neg_scores[idx].item()
                if 0.8 < n_score < 0.95:
                    augmented_rows.append({
                        "query": query,
                        "positive": entry['positive'],  # Old Positive
                        "negative": aug_negatives[idx], # New Negative
                        "meta": "aug_neg"
                    })

        # Combine Original + Augmented
        print(f"Generated {len(augmented_rows)} new rows from {len(json_data)} original rows.")
        return json_data + augmented_rows

# --- Execution ---
if __name__ == "__main__":
    # Sample Data (Triplet format)
    raw_data = [
        {
            "query": "best running shoes",
            "positive": "The Nike Pegasus is highly rated for jogging.",
            "negative": "How to install python on mac",
        },
        {
            "query": "symptoms of flu",
            "positive": "Influenza often causes high fever and body aches.",
            "negative": "The stock market is crashing today",
        }
    ]

    # Initialize
    # WARNING: Switch to "t5-base" if you do not have >12GB GPU memory
    augmenter = DataAugmenter(t5_model_name="google-t5/t5-3b")
    
    # Run
    final_dataset = augmenter.augment_dataset(raw_data, batch_size=2)
    
    # Save
    with open("augmented_triplets.json", "w") as f:
        json.dump(final_dataset, f, indent=4)
        
    # Preview
    print("\n--- Augmentation Preview ---")
    for row in final_dataset:
        if "meta" in row: # Print only new rows
            print(f"Type: {row['meta']}")
            print(f"Q: {row['query']}")
            print(f"P: {row['positive']}")
            print(f"N: {row['negative']}")
            print("-" * 20)