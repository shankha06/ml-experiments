import json
import numpy as np
import random
import pandas as pd

# from fuzzy import DMetaphone

# A simple QWERTY keyboard layout map for nearby keys
KEYBOARD_LAYOUT = {
    'q': ['w', 'a'], 'w': ['q', 'e', 's', 'a'], 'e': ['w', 'r', 'd', 's'],
    'r': ['e', 't', 'f', 'd'], 't': ['r', 'y', 'g', 'f'], 'y': ['t', 'u', 'h', 'g'],
    'u': ['y', 'i', 'j', 'h'], 'i': ['u', 'o', 'k', 'j'], 'o': ['i', 'p', 'l', 'k'],
    'p': ['o', 'l'],
    'a': ['q', 'w', 's', 'z'], 's': ['a', 'w', 'e', 'd', 'x', 'z'], 'd': ['s', 'e', 'r', 'f', 'c', 'x'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'], 'g': ['f', 't', 'y', 'h', 'b', 'v'], 'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'j': ['h', 'u', 'i', 'k', 'm', 'n'], 'k': ['j', 'i', 'o', 'l', 'm'], 'l': ['k', 'o', 'p'],
    'z': ['a', 's', 'x'], 'x': ['z', 's', 'd', 'c'], 'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'], 'b': ['v', 'g', 'h', 'n'], 'n': ['b', 'h', 'j', 'm'],
    'm': ['n', 'j', 'k']
}

def introduce_typos(text, num_errors=1):
    """
    Introduces a specified number of random character-level errors.
    
    Args:
        text (str): The original string.
        num_errors (int): The number of errors to introduce.
    
    Returns:
        str: The string with typos.
    """
    if len(text) <= num_errors:
        return text  # Avoid an infinite loop on small strings

    # Ensure we don't introduce errors at the same index
    indices_to_change = random.sample(range(len(text)), num_errors)
    
    new_text = list(text)
    for i in indices_to_change:
        rand_val = random.random()
        
        # Levenshtein-style errors
        if rand_val < 0.4:
            # Deletion: 40% chance
            new_text.pop(i)
        elif rand_val < 0.8:
            # Insertion: 40% chance
            new_text.insert(i, random.choice('abcdefghijklmnopqrstuvwxyz'))
        else:
            # Substitution: 20% chance
            new_text[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
            
    return "".join(new_text)

def introduce_keyboard_errors(text):
    """
    Simulates errors based on key proximity on a QWERTY keyboard.
    
    Args:
        text (str): The original string.
        
    Returns:
        str: The string with keyboard-based typos.
    """
    text = text.lower()
    if not text:
        return text
    
    char_to_change_idx = random.randint(0, len(text) - 1)
    char_to_change = text[char_to_change_idx]
    
    if char_to_change in KEYBOARD_LAYOUT:
        # Get a random nearby key
        new_char = random.choice(KEYBOARD_LAYOUT[char_to_change])
        new_text = list(text)
        new_text[char_to_change_idx] = new_char
        return "".join(new_text)
    
    return text  # No change if the character isn't in our map

def introduce_phonetic_errors(text):
    """
    Generates generic phonetic misspellings for a given word.
    
    This function uses a simplified, rule-based approach to introduce common 
    phonetic errors (e.g., substituting 's' for 'c', 'ph' for 'f', 'ie' for 'y').
    It processes one word at a time and applies a random substitution if
    a match is found. This is a heuristic approach and doesn't rely on
    a phonetic algorithm like Double Metaphone.
    
    Args:
        text (str): The original string.
        
    Returns:
        str: The string with a potential phonetic misspelling or the original
             text if no substitution is made.
    """
    # Define a dictionary of common phonetic substitutions.
    # The keys are substrings to look for, and the values are lists of
    # their phonetic replacements.
    phonetic_substitutions = {
        'ph': ['f', 'v'],
        'ie': ['y', 'i'],
        'ei': ['i', 'y'],
        'tion': ['shun', 'chun'],
        'c': ['s', 'k'],
        's': ['c', 'z'],
        'oo': ['u'],
        'ou': ['ow'],
        'th': ['t', 'd'],
        'ea': ['i', 'e']
    }
    
    words = text.split()
    modified_words = []
    
    for word in words:
        original_word = word
        lower_word = word.lower()
        
        # We'll randomly select one substitution to apply to the word.
        found_substitution = False
        keys_to_check = list(phonetic_substitutions.keys())
        random.shuffle(keys_to_check)

        for key in keys_to_check:
            if key in lower_word:
                # Choose a random replacement from the list.
                replacement = random.choice(phonetic_substitutions[key])
                # Replace the first occurrence of the key.
                modified_word = lower_word.replace(key, replacement, 1)
                modified_words.append(modified_word)
                found_substitution = True
                break  # Apply only one substitution per word
        
        if not found_substitution:
            modified_words.append(original_word)
            
    return ' '.join(modified_words)


def generate_misspellings_for_catalog(catalog_items, num_variants=5):
    """
    Generates a dataset of misspellings for a list of catalog items.
    
    Args:
        catalog_items (list): A list of product names or categories.
        num_variants (int): Number of misspelled versions to create for each item.
    
    Returns:
        list: A list of dictionaries, each with 'misspelled' and 'correct' keys.
    """
    generated_data = []
    
    for item in catalog_items:
        clean_item = item.strip().lower()
        
        # Add the clean version to the dataset
        generated_data.append({"input": clean_item, "target": clean_item})
        
        # Generate multiple misspelled variants
        for _ in range(num_variants):
            
            # Randomly select a misspelling technique
            technique = random.choice([
                introduce_typos, 
                introduce_keyboard_errors,
                introduce_phonetic_errors
            ])
            
            misspelled_text = technique(clean_item)
            
            # Avoid adding the original text as a misspelling
            if misspelled_text != clean_item:
                generated_data.append({"input": misspelled_text, "target": clean_item})
                
    # Use a set to remove duplicates before returning
    unique_data = {tuple(d.items()) for d in generated_data}
    return [dict(t) for t in unique_data]

### 3. Example Usage with a Retail Catalog

# A sample of your product catalog
data = pd.read_excel("notebooks/taxonomy-with-ids.en-US.xlsx")
product_categories = data["category"].to_list()
product_categories = [x.lower().strip() for x in product_categories if len(str(x)) > 4]

# Generate synthetic training data
data = []

for i in range(2):
    print(f"Generating batch {i} of sample misspellings ...")
    synthetic_data = generate_misspellings_for_catalog(product_categories, num_variants=1000)
    for entry in synthetic_data:
        data.append(
            {"input": f"Correct the spelling: {entry['input']}", "target": entry['target']}
        )

# Print the generated dataset

for entry in synthetic_data[:20]:  # Print the first 20 entries as an example
    print(f"Misspelled: {entry['input']} -> Correct: {entry['target']}")

with open("notebooks/synthetic_misspellings.json", "w") as f:
    json.dump(data, f, indent=4)
