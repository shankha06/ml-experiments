import json
import random
from collections import defaultdict
from itertools import combinations

def create_triplets(data, num_triplets=10000, hard_mining_threshold=0.5):
    """
    Creates a JSON dataset of hard and semi-hard triplets for a given dataset.

    Args:
        data (dict): A dictionary mapping merchant_name to a list of tags.
        num_triplets (int): The number of triplets to generate.
        hard_mining_threshold (float): The minimum overlap ratio for a negative sample to be considered "hard".
                                      A higher value means harder triplets.

    Returns:
        list: A list of dictionaries, where each dictionary represents a triplet.
    """
    merchants = list(data.keys())
    triplets = []
    
    # Create a mapping from individual tags to the merchants that use them
    tag_to_merchants = defaultdict(set)
    for merchant, tags in data.items():
        for tag in tags:
            tag_to_merchants[tag].add(merchant)

    # Convert the tags_list to a set for faster lookups
    data_sets = {merchant: set(tags) for merchant, tags in data.items()}

    # --- Triplet Generation Loop ---
    for _ in range(num_triplets):
        # 1. Select a random anchor merchant and its tags (query)
        anchor_merchant = random.choice(merchants)
        anchor_tags = data_sets[anchor_merchant]
        
        # 2. Select a positive merchant and its tags
        # A simple way to find a positive is to find another merchant that shares some tags
        # In a real-world scenario, you might have a more sophisticated way to define positives
        other_merchants = [m for m in merchants if m != anchor_merchant]
        if not other_merchants:
            continue

        positive_merchant = None
        for _ in range(10): # Try a few times to find a good positive
            candidate_merchant = random.choice(other_merchants)
            candidate_tags = data_sets[candidate_merchant]
            
            # A simple rule for a positive: has a significant overlap with the anchor
            if len(anchor_tags.intersection(candidate_tags)) > 0.5 * len(anchor_tags):
                positive_merchant = candidate_merchant
                break
        
        if not positive_merchant:
            continue
        
        positive_tags = data_sets[positive_merchant]
        
        # 3. Find a hard negative merchant
        hard_negative_merchant = None
        for _ in range(100): # Try a few times to find a hard negative
            candidate_merchant = random.choice([m for m in merchants if m != anchor_merchant and m != positive_merchant])
            candidate_tags = data_sets[candidate_merchant]
            
            # --- Hard and Semi-Hard Mining Logic ---
            # A negative is "hard" if it has a high overlap with the positive tags
            overlap_ratio = len(positive_tags.intersection(candidate_tags)) / len(positive_tags)
            
            # The logic below can be adjusted for your specific needs
            if overlap_ratio >= hard_mining_threshold:
                hard_negative_merchant = candidate_merchant
                break
        
        if not hard_negative_merchant:
            continue

        hard_negative_tags = data_sets[hard_negative_merchant]
        
        # 4. Create the triplet dictionary
        triplet = {
            "query": list(anchor_tags),
            "similar_tags": list(positive_tags),
            "negative_tags": list(hard_negative_tags)
        }
        triplets.append(triplet)
        
    return triplets

# --- Example Usage ---
# Dummy data based on your description
data = {
    "merchant_A": ["food", "thai", "spicy", "restaurant"],
    "merchant_B": ["food", "sushi", "japanese", "restaurant"],
    "merchant_C": ["clothing", "menswear", "formal", "shirts"],
    "merchant_D": ["food", "indian", "curry", "spicy"],
    "merchant_E": ["clothing", "womenswear", "dresses"],
    "merchant_F": ["restaurant", "seafood", "fine dining"],
    "merchant_G": ["food", "korean", "bbq", "spicy"],
    "merchant_H": ["food", "mexican", "tacos", "burritos", "spicy"],
}

# Add more dummy data to reach ~10K tags and various merchants
# In a real scenario, this would be your loaded dataset
for i in range(100):
    merchant_name = f"merchant_{i+10}"
    tags = random.sample(list(set(sum(data.values(), []))), random.randint(2, 5))
    data[merchant_name] = tags

# Generate the triplets
triplets_data = create_triplets(data, num_triplets=1000)

# Save to a JSON file
with open('triplets_dataset.json', 'w') as f:
    json.dump(triplets_data, f, indent=4)

print(f"Generated {len(triplets_data)} triplets and saved to triplets_dataset.json")