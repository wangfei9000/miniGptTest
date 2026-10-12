import torch
import tiktoken
import sys
from model import GPTLanguageModel, device, remap_tokens, custom_decode

# Load mappings and vocab size
old_to_new = torch.load('data/old_to_new.pt', map_location=device)
new_to_old = torch.load('data/new_to_old.pt', map_location=device)
vocab_size = len(old_to_new)

# Tokenizer
enc = tiktoken.get_encoding("gpt2")

# Load model
model = GPTLanguageModel(vocab_size)
checkpoint = torch.load('MiniStoryGPT-30M.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Get prompt from command line or default
if len(sys.argv) > 1:
    prompt = ' '.join(sys.argv[1:])
else:
    prompt = "Once upon a time,"

print(f"Generating story starting with: '{prompt}'")

original_context = enc.encode(prompt)
remapped_context = remap_tokens(original_context, old_to_new)
context = torch.tensor([remapped_context], dtype=torch.long, device=device)

with torch.no_grad():
    output = model.generate(context, max_new_tokens=300)
    story = custom_decode(output[0].tolist(), new_to_old, enc)
    print(story)