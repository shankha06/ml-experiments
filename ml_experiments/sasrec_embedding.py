
import torch
import torch.nn as nn


class SASRecModule(nn.Module):
    def __init__(
        self, num_features, embedding_dim, num_heads, num_layers, max_len, dropout=0.1
    ):
        super(SASRecModule, self).__init__()

        self.embedding_dim = embedding_dim

        # 1. Embeds the feature vector of each month into a dense vector
        self.month_embedding = nn.Linear(num_features, embedding_dim)

        # 2. Adds positional information to the sequence
        self.positional_embedding = nn.Embedding(max_len, embedding_dim)

        # 3. The core Self-Attention block
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True,  # Crucial for our data shape
        )

        norm = nn.LayerNorm(embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=num_layers, norm=norm
        )

        # 4. Predicts the next month's feature vector
        self.prediction_layer = nn.Linear(embedding_dim, num_features)

        self.dropout = nn.Dropout(dropout)

    def _generate_causal_mask(self, sz):
        # Creates a mask to ensure the model doesn't cheat by looking ahead
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, 0.0)
        )
        return mask

    def forward(self, input_sequence):
        # input_sequence shape: (batch_size, seq_len, num_features)
        batch_size, seq_len, _ = input_sequence.shape
        device = input_sequence.device

        # Create positional IDs (0, 1, 2, ...)
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        # Apply embeddings
        month_embeds = self.month_embedding(input_sequence)
        pos_embeds = self.positional_embedding(position_ids)

        # Combine month and positional embeddings
        x = self.dropout(month_embeds + pos_embeds)

        # Create the causal attention mask
        causal_mask = self._generate_causal_mask(seq_len).to(device)

        # Pass through the transformer encoder
        transformer_output = self.transformer_encoder(x, mask=causal_mask)

        # We only want to predict the next step based on the *last* element of the input sequence
        last_element_output = transformer_output[
            :, -1, :
        ]  # (batch_size, embedding_dim)

        # Predict the next month's feature vector
        predicted_next_month = self.prediction_layer(
            last_element_output
        )  # (batch_size, num_features)

        return predicted_next_month


class EarlyStopping:
    """
    Simple early stopping:
    - Stops training if eval loss hasn't improved by min_delta for `patience` consecutive epochs.
    - Stores and restores the best model state_dict.
    """

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")
        self.best_state = None
        self.best_epoch = -1

    def step(self, metric, model, epoch_idx):
        improved = metric < (self.best - self.min_delta)
        if improved:
            self.best = metric
            self.counter = 0
            # Store best state on CPU to minimize GPU memory usage
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self.best_epoch = epoch_idx
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# --- Configuration ---
NUM_USERS = 10000
SEQUENCE_LENGTH = 8  # Total months of data per user
NUM_FEATURES = 144  # Example: Number of features per month
EMBEDDING_DIM = 128  # Model's internal dimension
NUM_HEADS = 4  # Number of attention heads
NUM_LAYERS = 8  # Number of transformer blocks
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Early stopping configuration
PATIENCE = 3
MIN_DELTA = 1e-4

# Device configuration: Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Simulate User Data ---
# Shape: (num_users, sequence_length, num_features)
user_data = torch.randn(NUM_USERS, SEQUENCE_LENGTH+4, NUM_FEATURES)
print(f"Simulated data shape: {user_data.shape}\n")


# --- 2. Create Training Examples ---
# We will create input/target pairs for next-month prediction
inputs = []
targets = []

for user_seq in user_data:
    # We create sequences of increasing length to train the model
    for i in range(1, SEQUENCE_LENGTH):
        # Input is from the beginning up to month `i`
        input_sub_seq = user_seq[:i, :]

        # Target is the very next month `i`
        target_month = user_seq[i, :]

        # We need to pad inputs to a fixed length for batching
        padded_input = torch.zeros(SEQUENCE_LENGTH - 1, NUM_FEATURES)
        padded_input[:i, :] = input_sub_seq

        inputs.append(padded_input)
        targets.append(target_month)

# Convert to tensors
inputs = torch.stack(inputs)
targets = torch.stack(targets)

# Create Train/Eval splits and DataLoaders
VAL_SPLIT = 0.2  # 20% of the data for evaluation

dataset = torch.utils.data.TensorDataset(inputs, targets)
num_samples = len(dataset)
eval_size = max(1, int(VAL_SPLIT * num_samples))
train_size = num_samples - eval_size

# Reproducible split
generator = torch.Generator().manual_seed(42)
train_dataset, eval_dataset = torch.utils.data.random_split(
    dataset, [train_size, eval_size], generator=generator
)

# Use pin_memory for faster GPU transfer if using CUDA
pin = device.type == "cuda"
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin
)
eval_loader = torch.utils.data.DataLoader(
    eval_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin
)

print(f"Train examples: {len(train_dataset)}, Eval examples: {len(eval_dataset)}")

print(f"Total training examples: {len(inputs)}")
print(f"Example input shape: {inputs[0].shape}")
print(f"Example target shape: {targets[0].shape}\n")


# --- 3. Initialize Model, Loss, and Optimizer ---
model = SASRecModule(
    num_features=NUM_FEATURES,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_len=SEQUENCE_LENGTH - 1,
).to(device)

# Since we are predicting a continuous feature vector, MSE is a good choice
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

early_stopper = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)


# --- Training loop with per-epoch Train/Eval loss and Early Stopping ---
print("--- Starting Training ---")
for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss_sum = 0.0
    train_batches = 0

    for batch_inputs, batch_targets in train_loader:
        # Move data to device
        batch_inputs = batch_inputs.to(device, non_blocking=pin)
        batch_targets = batch_targets.to(device, non_blocking=pin)

        optimizer.zero_grad()
        # Forward pass (adjust if your model requires additional inputs)
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        train_batches += 1

    avg_train_loss = train_loss_sum / max(1, train_batches)

    # Eval
    model.eval()
    eval_loss_sum = 0.0
    eval_batches = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in eval_loader:
            batch_inputs = batch_inputs.to(device, non_blocking=pin)
            batch_targets = batch_targets.to(device, non_blocking=pin)

            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            eval_loss_sum += loss.item()
            eval_batches += 1

    avg_eval_loss = eval_loss_sum / max(1, eval_batches)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}"
    )

    # Early stopping check
    should_stop = early_stopper.step(avg_eval_loss, model, epoch_idx=epoch)
    if should_stop:
        print(
            f"Early stopping triggered at epoch {epoch+1}. Best eval loss: {early_stopper.best:.4f} (epoch {early_stopper.best_epoch+1})"
        )
        break

# Restore best model (if early stopping saved one)
early_stopper.restore_best(model)

print("--- Training Complete ---")

# --- 5. How to Get Embeddings (Inference) ---
# After training, you can get a user's embedding by feeding their data
# into the model but stopping before the final prediction layer.

# Example: Get embedding for the first user's first 5 months
model.eval()
with torch.no_grad():
    example_user_seq = user_data[0:1, :5, :].to(device)  # Shape: (1, 5, 25)

    # Internal representation up to Transformer encoder
    month_embeds = model.month_embedding(example_user_seq)
    pos_ids = torch.arange(5, device=device).unsqueeze(0)
    pos_embeds = model.positional_embedding(pos_ids)
    x = model.dropout(month_embeds + pos_embeds)
    mask = model._generate_causal_mask(5).to(device)

    transformer_output = model.transformer_encoder(x, mask=mask)

    # The embedding is the output for the last element in the sequence
    user_embedding = transformer_output[:, -1, :]

    print(f"\nShape of the final user embedding: {user_embedding.shape}")
