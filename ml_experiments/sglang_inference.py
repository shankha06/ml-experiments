# python3 -m sglang.launch_server --model-path openai/gpt-oss-20b --host 0.0.0.0 --port 30000 --trust-remote-code

import os
from openai import OpenAI

# Set up the OpenAI client to point to the SGLang server
# Replace the base_url with the address and port of your SGLang server.
client = OpenAI(
    api_key="EMPTY",  # API key is not required for the local SGLang server
    base_url="http://localhost:30000/v1",
)

def run_inference(prompt, model_name="openai/gpt-oss-20b"):
    """
    Sends a prompt to the SGLang server for inference and prints the response.
    """
    messages = [
        {"role": "system", "content": "You are a helpful and creative assistant."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=256,
            temperature=0.7,
            stream=False,  # Set to True for streaming responses
        )
        # Extract and print the generated text
        if response.choices:
            generated_text = response.choices[0].message.content
            print(f"Generated Text:\n{generated_text}")
        else:
            print("No response choices found.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    user_prompt = "Explain the concept of quantum entanglement in simple terms."
    print(f"User Prompt: {user_prompt}\n")
    run_inference(user_prompt)