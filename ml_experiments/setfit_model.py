import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sklearn.model_selection import train_test_split

def train_setfit_classifier():
    # ---------------------------------------------------------
    # 1. PREPARE DATA
    # ---------------------------------------------------------
    # For demonstration, we create a dummy dataset matching your structure.
    # In production, you would load your file: df = pd.read_csv("your_data.csv")
    
    data = {
        "question": [
            "How do I reset my password?",
            "What is the refund policy?",
            "My screen is frozen, what do I do?",
            "Where can I track my order?",
            "I received a damaged item.",
            "Can I change my username?",
            "Do you offer technical support?",
            "How long does shipping take?",
            "The app keeps crashing.",
            "I want to return a purchase."
        ],
        "class": [
            "Account", 
            "Billing", 
            "Technical", 
            "Shipping", 
            "Shipping", 
            "Account", 
            "Technical", 
            "Shipping", 
            "Technical", 
            "Billing"
        ]
    }

    df = pd.DataFrame(data)
    
    # Split into train and evaluation sets
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")

    # ---------------------------------------------------------
    # 2. INITIALIZE MODEL
    # ---------------------------------------------------------
    # We use a small, fast Sentence Transformer model.
    # You can change this to 'sentence-transformers/paraphrase-mpnet-base-v2' for higher accuracy.
    model_id = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    
    print(f"\nDownloading and initializing model: {model_id}...")
    model = SetFitModel.from_pretrained(model_id)

    # ---------------------------------------------------------
    # 3. CONFIGURE TRAINER
    # ---------------------------------------------------------
    # CRITICAL STEP: mapping your custom columns to what SetFit expects.
    # "question" maps to SetFit's "text" input
    # "class" maps to SetFit's "label" target
    column_mapping = {
        "question": "text",
        "class": "label"
    }

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_class=None,          # Uses default CosineSimilarityLoss
        metric="accuracy",        # Metric to evaluate on
        batch_size=16,
        num_iterations=20,        # Number of text pairs to generate for contrastive learning
        num_epochs=1,             # One epoch is often enough for SetFit
        column_mapping=column_mapping 
    )

    # ---------------------------------------------------------
    # 4. TRAIN AND EVALUATE
    # ---------------------------------------------------------
    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating model...")
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")

    # ---------------------------------------------------------
    # 5. INFERENCE CHECK
    # ---------------------------------------------------------
    # Let's test it on a new unseen question
    test_questions = [
        "I forgot my login credentials.",
        "When will my package arrive?"
    ]
    
    print("\nRunning inference on test questions:")
    preds = model.predict(test_questions)
    
    for q, p in zip(test_questions, preds):
        print(f"Question: '{q}' -> Predicted Class: {p}")

    # ---------------------------------------------------------
    # 6. SAVE MODEL
    # ---------------------------------------------------------
    save_path = "./my_setfit_model"
    model.save_pretrained(save_path)
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    # Ensure setfit is installed: pip install setfit
    train_setfit_classifier()