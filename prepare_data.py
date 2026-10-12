from collections import Counter
import torch
from datasets import load_dataset
import tiktoken
import os

def prepare_data():
    # Load Dataset
    dataset = load_dataset("roneneldan/TinyStories")
    
    # Tokenize with GPT 2 style BPE
    enc = tiktoken.get_encoding("gpt2")
    token_counter = Counter()
    
    # Process training data
    for text in dataset['train']['text']:
        tokens = enc.encode(text)
        token_counter.update(tokens)
        
    # Process validation data
    for text in dataset['validation']['text']:
        tokens = enc.encode(text)
        token_counter.update(tokens)
    
    # Get the top 10K most common tokens (sorted by frequency descending)
    top_tokens = [token for token, count in token_counter.most_common(10000)]
    
    # Create mappings: old token ID -> new token ID (0 to 9999), sorted by original ID for consistency
    sorted_top_tokens = sorted(top_tokens)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted_top_tokens)}
    
    # Reverse mapping for decoding: new ID -> old ID
    new_to_old = {new_id: old_id for old_id, new_id in old_to_new.items()}
    
    # New vocab size
    new_vocab_size = len(sorted_top_tokens)  # Should be 10000
    
    # Function to remap encoded tokens to new IDs
    def remap_tokens(encoded_tokens):
        return [old_to_new.get(token, 0) for token in encoded_tokens]  # Use 0 as UNK if not in top 10K
    
    # Re-encode the dataset using the new mapping
    train_tokens = []
    for text in dataset['train']['text']:
        original_tokens = enc.encode(text)
        remapped = remap_tokens(original_tokens)
        train_tokens.extend(remapped)
    
    val_tokens = []
    for text in dataset['validation']['text']:
        original_tokens = enc.encode(text)
        remapped = remap_tokens(original_tokens)
        val_tokens.extend(remapped)
    
    # Convert to tensors
    train_data = torch.tensor(train_tokens, dtype=torch.long)
    val_data = torch.tensor(val_tokens, dtype=torch.long)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save as binaries and mappings
    torch.save(train_data, 'data/train.bin')
    torch.save(val_data, 'data/val.bin')
    torch.save(old_to_new, 'data/old_to_new.pt')
    torch.save(new_to_old, 'data/new_to_old.pt')
    
    print(f"New vocab size: {new_vocab_size}")
    print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

if __name__ == "__main__":
    prepare_data()