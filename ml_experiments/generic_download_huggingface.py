from huggingface_hub import snapshot_download

# Specify the model ID from the Hugging Face Hub
repo_id = "openai/gpt-oss-20b"

# Specify the local directory to save the model files
local_dir = "./gpt-oss-20b_model"

# Download all files from the repository to the specified local directory
try:
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"✅ Successfully downloaded the model to {local_dir}")

except Exception as e:
    print(f"❌ An error occurred during download: {e}")