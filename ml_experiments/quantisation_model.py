import torch
import pandas as pd
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Define the model to be quantized
model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "Llama-2-7b-AWQ"

# --- Use your local data here ---
# Create a list of strings from your local text file.
data = pd.read_excel("data/sample_data.xlsx")
local_data = data["text"].tolist()


# Quantization configuration
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Load the model and tokenizer
model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True})
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize the model using your local data
model.quantize(tokenizer, quant_config=quant_config, calibration_data=local_data)

# Save the quantized model
model.save_quantized(quant_path, safetensors=True)
tokenizer.save_pretrained(quant_path)

print("Model quantized using local data.")