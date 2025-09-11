import random
import time
import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
import random
import pandas as pd
from faker import Faker

def generate_synthetic_ner_data(num_samples=100):
    """Generates synthetic NER training data for merchant and location entities.

    Args:
        num_samples (int): The number of synthetic data samples to generate.

    Returns:
        list: A list of spaCy-formatted training examples.
    """
    fake = Faker()
    training_data = []

    data = pd.read_excel("data/taxonomy-with-ids.en-US.xlsx")
    product_categories = data["category"].to_list()
    product_categories = [x.lower().strip() for x in product_categories if len(str(x)) > 4]

    # Simple templates to embed entities
    templates = [
        "nearest {} deals to {}",
        "{} near {}",
        "{} near me",
        "{} deals",
        "offers for {} in {}",
        "{} in {}",
        "{} discounts",
        "deals for {}",
        "{}",
        "discounts for {}",
        "{} offers",
        "offers for {}",
    ]

    # Generate a list of fake merchants and locations for diversity
    merchants = [fake.company() for _ in range(5000)]
    locations = [fake.city() for _ in range(5000)]
    merchants.extend(product_categories)

    for _ in range(num_samples):
        # Choose a random template
        template = random.choice(templates)
        
        # Pick a random merchant and location
        merchant_name = random.choice(merchants)
        location_name = random.choice(locations)
        
        # Determine which entities to use in the template
        entities_in_query = []
        if "{}" in template:
            entities_in_query.append(('MERCHANT', merchant_name))
        if "{}" in template:
            entities_in_query.append(('LOCATION', location_name))

        # Fill the template with the chosen entities
        if len(entities_in_query) == 2:
            text = template.format(merchant_name, location_name)
        elif len(entities_in_query) == 1:
            text = template.format(merchant_name)
        else:
            continue

        # Create the NER annotations (start and end indices)
        entities = []
        current_text = text
        
        # Find and annotate the merchant name
        if 'MERCHANT' in [e[0] for e in entities_in_query]:
            start = text.find(merchant_name)
            if start != -1:
                end = start + len(merchant_name)
                entities.append((start, end, 'MERCHANT'))

        # Find and annotate the location name
        if 'LOCATION' in [e[0] for e in entities_in_query]:
            start = text.find(location_name)
            if start != -1:
                end = start + len(location_name)
                entities.append((start, end, 'LOCATION'))
                
        # Append the annotated example to our dataset
        training_data.append((text.lower(), {"entities": entities}))

    return training_data

def train_ner_model(training_data, model_name="ner_merchants", epochs=13, use_gpu=True):
    """
    Trains a custom NER model using the provided training data with GPU support.

    Args:
        training_data (list): A list of spaCy-formatted training examples.
        model_name (str): The name to save the trained model under.
        epochs (int): Number of training epochs.
        use_gpu (bool): Whether to use GPU for training if available.
    """
    # Check for GPU availability and configure spaCy
    gpu_id = -1  # Default to CPU
    if use_gpu:
        try:
            # Try to use GPU
            spacy.prefer_gpu()
            if spacy.util.is_package("en_core_web_sm"):
                # If you have a pre-trained model, you can check GPU availability
                import torch
                if torch.cuda.is_available():
                    gpu_id = 0
                    print(f"GPU detected and will be used for training (GPU ID: {gpu_id})")
                else:
                    print("GPU requested but not available. Using CPU instead.")
            else:
                print("GPU requested but PyTorch not available. Using CPU instead.")
        except Exception as e:
            print(f"Error setting up GPU: {e}. Falling back to CPU.")
    else:
        print("Using CPU for training (GPU disabled by user).")
    
    # Create a blank English language model
    nlp = spacy.blank("en")
    
    # Configure GPU if available
    if gpu_id >= 0:
        try:
            spacy.require_gpu(gpu_id)
            print(f"Successfully configured spaCy to use GPU {gpu_id}")
        except Exception as e:
            print(f"Failed to configure GPU: {e}. Falling back to CPU.")
            gpu_id = -1
    
    # Add the NER component to the pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add the custom labels to the NER component
    ner.add_label("MERCHANT")
    ner.add_label("LOCATION")
    
    # Disable other pipeline components during training to only train NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        # Initialize the optimizer
        optimizer = nlp.begin_training()
        
        # Configure batch size based on available memory (larger batches for GPU)
        batch_size = 64 if gpu_id >= 0 else 16
        
        print(f"Starting training with batch size: {batch_size}")
        print(f"Training device: {'GPU' if gpu_id >= 0 else 'CPU'}")
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            losses = {}
            
            # Process data in batches for better GPU utilization
            batches = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]
            
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    # Create an Example object for training
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                
                # Update the model with the batch
                nlp.update(examples, drop=0.5, losses=losses)
            
            print(f"Epoch {epoch+1}/{epochs} Loss: {losses['ner']:.4f}")

    # Save the trained model to disk
    nlp.to_disk(f"{model_name}")
    print(f"Model saved to {model_name}")
    
    # Print final training summary
    device_used = "GPU" if gpu_id >= 0 else "CPU"
    print(f"Training completed using {device_used}")
    print(f"Total training samples: {len(training_data)}")
    print(f"Batch size used: {batch_size}")


def evaluate_ner_model(nlp_model, eval_data):
    """
    Evaluates the NER model performance using evaluation data.
    
    Args:
        nlp_model: Trained spaCy NLP model
        eval_data: List of tuples (text, annotations) for evaluation
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    true_positives = {'MERCHANT': 0, 'LOCATION': 0}
    false_positives = {'MERCHANT': 0, 'LOCATION': 0}
    false_negatives = {'MERCHANT': 0, 'LOCATION': 0}
    
    total_predictions = 0
    total_ground_truth = 0
    
    for text, annotations in eval_data:
        # Get predictions from the model
        doc = nlp_model(text)
        predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        
        # Get ground truth entities
        ground_truth_entities = annotations['entities']
        
        total_predictions += len(predicted_entities)
        total_ground_truth += len(ground_truth_entities)
        
        # Convert to sets for easier comparison
        pred_set = set(predicted_entities)
        truth_set = set(ground_truth_entities)
        
        # Calculate true positives (exact matches)
        for entity in pred_set.intersection(truth_set):
            label = entity[2]
            true_positives[label] += 1
        
        # Calculate false positives (predicted but not in ground truth)
        for entity in pred_set - truth_set:
            label = entity[2]
            false_positives[label] += 1
        
        # Calculate false negatives (in ground truth but not predicted)
        for entity in truth_set - pred_set:
            label = entity[2]
            false_negatives[label] += 1
    
    # Calculate metrics for each entity type
    metrics = {}
    overall_tp = sum(true_positives.values())
    overall_fp = sum(false_positives.values())
    overall_fn = sum(false_negatives.values())
    
    for label in ['MERCHANT', 'LOCATION']:
        tp = true_positives[label]
        fp = false_positives[label]
        fn = false_negatives[label]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    # Calculate overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    metrics['overall'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1,
        'true_positives': overall_tp,
        'false_positives': overall_fp,
        'false_negatives': overall_fn,
        'total_predictions': total_predictions,
        'total_ground_truth': total_ground_truth
    }
    
    return metrics

def print_evaluation_results(metrics):
    """
    Prints formatted evaluation results.
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print("=" * 60)
    print("NER MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    # Print per-entity metrics
    for label in ['MERCHANT', 'LOCATION']:
        print(f"\n{label} Entity:")
        print(f"  Precision: {metrics[label]['precision']:.4f}")
        print(f"  Recall:    {metrics[label]['recall']:.4f}")
        print(f"  F1-Score:  {metrics[label]['f1_score']:.4f}")
        print(f"  True Positives:  {metrics[label]['true_positives']}")
        print(f"  False Positives: {metrics[label]['false_positives']}")
        print(f"  False Negatives: {metrics[label]['false_negatives']}")
    
    # Print overall metrics
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Precision: {metrics['overall']['precision']:.4f}")
    print(f"  Recall:    {metrics['overall']['recall']:.4f}")
    print(f"  F1-Score:  {metrics['overall']['f1_score']:.4f}")
    print(f"  Total Predictions: {metrics['overall']['total_predictions']}")
    print(f"  Total Ground Truth: {metrics['overall']['total_ground_truth']}")
    print("=" * 60)

def analyze_errors(nlp_model, eval_data, num_examples=10):
    """
    Analyzes and displays prediction errors for debugging.
    
    Args:
        nlp_model: Trained spaCy NLP model
        eval_data: List of tuples (text, annotations) for evaluation
        num_examples: Number of error examples to display
    """
    print("\nERROR ANALYSIS:")
    print("-" * 40)
    
    error_count = 0
    for text, annotations in eval_data:
        if error_count >= num_examples:
            break
            
        doc = nlp_model(text)
        predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        ground_truth_entities = annotations['entities']
        
        pred_set = set(predicted_entities)
        truth_set = set(ground_truth_entities)
        
        # Check if there are any errors (mismatches)
        if pred_set != truth_set:
            error_count += 1
            print(f"\nExample {error_count}:")
            print(f"Text: '{text}'")
            print(f"Ground Truth: {ground_truth_entities}")
            print(f"Predicted:    {predicted_entities}")
            
            # Show specific error types
            false_positives = pred_set - truth_set
            false_negatives = truth_set - pred_set
            
            if false_positives:
                print(f"False Positives: {list(false_positives)}")
            if false_negatives:
                print(f"False Negatives: {list(false_negatives)}")

if __name__ == '__main__':
    # Generate a large dataset for training
    synthetic_data = generate_synthetic_ner_data(num_samples=9000)
    print(synthetic_data[:5])

    MODEL_NAME = "models/custom_ner"
    # Split data into training and validation sets
    random.shuffle(synthetic_data)
    split_point = int(len(synthetic_data) * 0.8)
    train_data = synthetic_data[:split_point]
    # You would use eval_data for evaluation, but this example focuses on training
    eval_data = synthetic_data[split_point:]

    # Train the model
    train_ner_model(train_data, model_name=MODEL_NAME, epochs=8, use_gpu=True)

    # --- Test the trained model ---

    # Load the trained model
    nlp_custom = spacy.load(MODEL_NAME)
    
    test_query = "deals for burgers in San Francisco"
    doc = nlp_custom(test_query)

    print(f"\nTest Query: {test_query}")
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")

    # --- Evaluate the trained model ---

    # Evaluate the model
    start_time = time.time()
    evaluation_metrics = evaluate_ner_model(nlp_custom, eval_data)
    end_time = time.time()

    print(f"\nEvaluation Time: {(end_time - start_time)/len(eval_data):.2f} seconds")
    
    # Print results
    print_evaluation_results(evaluation_metrics)
    
    # Analyze errors
    analyze_errors(nlp_custom, eval_data, num_examples=5)