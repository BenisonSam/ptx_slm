import os
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Configuration
PERFORMANCE_MODE = True  # Set to True for maximum performance, False for memory optimization
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Create a directory name from the model name (replacing / with _)
model_dir = os.path.join("models", re.sub(r'[^a-zA-Z0-9\-/]', '_', model_name))
model_dir = os.path.abspath(model_dir)

# Create models directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Check if model files exist
model_path = os.path.join(model_dir, "model")
tokenizer_path = os.path.join(model_dir, "tokenizer")

# Determine device
device = torch.device("cuda" if PERFORMANCE_MODE and torch.cuda.is_available() else "cpu")

# Default model kwargs and generation kwargs
model_kwargs = {
    "torch_dtype": torch.float16,  # Use float16 for GPU
    "device_map": "auto",  # Let transformers handle device placement
    "low_cpu_mem_usage": True,
}
generation_kwargs = {
    "do_sample": False,
    "num_beams": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.0,
    "use_cache": True,
}

# Determine device and optimization settings based on performance mode and available resources
if PERFORMANCE_MODE and torch.cuda.is_available():
    print("Running in high-performance mode with GPU acceleration")
else:
    print("Running in memory-optimized mode on CPU")
    model_kwargs["torch_dtype"] = torch.float32  # Use float32 for CPU
    model_kwargs["device_map"] = None  # Disable device_map for CPU

# Download and save model if it doesn't exist
if not os.path.exists(model_path):
    print(f"Downloading model to {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    if device.type == "cpu":
        model = model.to(device)
    model.save_pretrained(model_path)
else:
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )
    if device.type == "cpu":
        model = model.to(device)

# Download and save tokenizer if it doesn't exist
if not os.path.exists(tokenizer_path):
    print(f"Downloading tokenizer to {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tokenizer_path)
else:
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

prompt = "What is the role of textual criticism?"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Can you explain what textual criticism is?"},
    {"role": "assistant", "content": "Textual criticism is the study of manuscripts and their variations to determine the most accurate version of a text. It's particularly important in biblical studies."},
    {"role": "user", "content": "What are its main goals?"},
    {"role": "assistant", "content": "The main goals of textual criticism are to identify and correct errors in texts, reconstruct the original text, and understand the history of its transmission."},
    {"role": "user", "content": "Name me some of the most important textual critics."},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# Skip the prompt
skip_prompt=False

# Create a streamer for token-by-token output
streamer = TextIteratorStreamer(tokenizer, skip_prompt=skip_prompt)

# Get the special token from tokenizer
eos_token = tokenizer.eos_token
print(f"EOS token: {eos_token}")

# Start timing
start_time = time.time()

# Start the generation in a separate thread
generation_kwargs = dict(
    **model_inputs,
    max_new_tokens=512,
    streamer=streamer,
    **generation_kwargs
)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Process the streamed output
generated_text = ""
for new_text in streamer:
    if skip_prompt and new_text.endswith(eos_token):
        new_text = new_text[:-len(eos_token)]
    if not skip_prompt or new_text != eos_token:
        print(new_text, end="", flush=True)
        generated_text += new_text

thread.join()

# Calculate generation metrics
end_time = time.time()
generation_time = end_time - start_time
num_tokens = len(tokenizer.encode(generated_text))
tokens_per_second = num_tokens / generation_time

print(f"\nGeneration completed in {generation_time:.2f} seconds")
print(f"Generated {num_tokens} tokens")
print(f"Tokens per second: {tokens_per_second:.2f}")
