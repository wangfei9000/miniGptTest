import torch
import csv
import os
import tiktoken
from model import GPTLanguageModel, device, batch_size, block_size, eval_iters

# Training hyperparameters
max_iters = 100000  # Training iterations
eval_interval = 500
learning_rate = 3e-4

version = "v1"
report_file = f"training_report_{version}.txt"
model_file = f"model_{version}.pth"
losses_file = f"losses_{version}.csv"

print(f"Training with {device}")

# Load data and mappings
old_to_new = torch.load('data/old_to_new.pt', map_location=device)
new_to_old = torch.load('data/new_to_old.pt', map_location=device)
train_data = torch.load('data/train.bin', map_location=device)
val_data = torch.load('data/val.bin', map_location=device)

vocab_size = len(old_to_new)

print(f"Loaded vocab size: {vocab_size}")
print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

# Tokenizer
enc = tiktoken.get_encoding("gpt2")

# Data loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Instantiate and move to device
model = GPTLanguageModel(vocab_size)
model = model.to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

# Check for checkpoint to load
start_iter = 0
checkpoint_path = None
for potential_path in sorted([f for f in os.listdir('.') if f.startswith('checkpoint_iter') and f.endswith('.pth')], reverse=True):
    if os.path.exists(potential_path):
        checkpoint_path = potential_path
        break

if checkpoint_path:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_iter = checkpoint['iter']
    print(f"Loaded checkpoint from {checkpoint_path} at iteration {start_iter}")
else:
    print("No checkpoint found, starting from scratch")

# Optimizer (load state for resuming)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
if checkpoint_path and start_iter > 0:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

losses_log = []

# Training loop
checkpoint_interval = 5000
sample_interval = eval_interval
original_context = enc.encode("Once upon a time,")
remapped_context = [old_to_new.get(token, 0) for token in original_context]
context = torch.tensor([remapped_context], dtype=torch.long, device=device)

for iter in range(start_iter, max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        with open(report_file, 'a') as f:
            f.write(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}\n")
        losses_log.append([iter, losses['train'], losses['val']])  # Log for plotting

    if iter % sample_interval == 0:  # Generate sample every x iterations
        print(f"\nSample story at iteration {iter}:")
        with torch.no_grad():
            output = model.generate(context, max_new_tokens=300)
            story = enc.decode([new_to_old.get(new_id, 0) for new_id in output[0].tolist()])
            print(story)
        print("-" * 50) 

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if iter % checkpoint_interval == 0 and iter > 0:
        checkpoint = {
            'iter': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }
        torch.save(checkpoint, f'checkpoint_iter{iter}.pth')
        print(f"Saved checkpoint at iteration {iter}")

# Save final model
torch.save(model.state_dict(), model_file)

with open(losses_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Iteration', 'Train Loss', 'Val Loss'])
    writer.writerows(losses_log)

print(f"\nSaved report to {report_file}, model to {model_file}, losses to {losses_file}")