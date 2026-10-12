# MiniStoryGPT

MiniStoryGPT is a compact educational implementation of a GPT-style language model developed to learn and demonstrate training transformer architectures from scratch. The project reproduces core ideas from "Attention is All You Need" and practical guidance from Andrej Karpathy's "Zero to Hero" materials and nanoGPT, and trains on the TinyStories dataset to generate short, child-friendly narratives.

HuggingFace Model Card: https://huggingface.co/johnnycwatt/MiniStoryGPT

**Purpose:** This repository is explicitly for educational and experimentation purposes — to gain hands-on experience building, training, and sampling from GPT-like models. It is not intended as a production-ready language model.

## Highlights

- Implements a small GPT language model in PyTorch.
- Uses the TinyStories dataset to train a storytelling model.
- Includes utilities for token remapping and custom decoding to support a reduced tokenizer/vocabulary.

## Results

The trained model included with this repository (see `MiniStoryGPT-30M.pth`) contains approximately 30 million parameters and was trained on a subset of TinyStories examples. On the training subset used, the model reached a training loss of approximately 1.55 after ~50,000 iterations.

## Repository Contents

- `model.py` — Model definition and helper functions (token remapping, device handling, custom decode, and `GPTLanguageModel`).
- `train.py` — Training loop and checkpoint saving (used to train `MiniStoryGPT-30M.pth`).
- `generate.py` — Simple script / example for loading the model and generating stories.
- `prepare_data.py` — Data preparation utilities to create the binary training files and token mapping files.
- `data/` — Preprocessed data and mapping files used by the notebook and scripts:
  - `train.bin`, `val.bin` — tokenized binary datasets (generated through the prepare_data.py script)
  - `old_to_new.pt`, `new_to_old.pt` — token remapping dictionaries saved as PyTorch tensors
- `MiniStoryGPT-30M.pth` — Example trained checkpoint (included for convenience). 50,000 Iters
- `notebook.ipynb` — Notebook walkthrough showing small-scale attention examples, model loading, sampling examples, and a project overview.
- `README.md` — This file.

## Quick Start

Requirements:

- Python 3.8+ (tested on 3.10)
- PyTorch
- `tiktoken` (for GPT-2 tokenization)
- `datasets` (optional, used in the notebook to inspect TinyStories)

Install dependencies (example):

```powershell
python -m pip install -r requirements.txt
```

If a `requirements.txt` is not present, install the main packages directly:

```powershell
python -m pip install torch tiktoken datasets
```

Download model and mappings from Hugging Face and generate:

1. Download the model checkpoint and mapping files from the Hugging Face repo `johnnycwatt/MiniStoryGPT`:

```powershell
# create a directory to store downloads
mkdir model_files;
cd model_files;

# download with git-lfs (recommended) or use the web UI to download files
git clone https://huggingface.co/johnnycwatt/MiniStoryGPT .
```

2. Ensure the following files are present in `model_files` and move them into your project `data/` or update paths accordingly:

- `pytorch_model.bin` or `MiniStoryGPT-30M.pth` — model checkpoint
- `old_to_new.pt` — token remapping dictionary
- `new_to_old.pt` — inverse token mapping

3. Generate using the project script:

```powershell
cd ..  # back to project root
python generate.py --checkpoint model_files/MiniStoryGPT-30M.pth
```

Or run the notebook `notebook.ipynb` (recommended for walkthrough and examples).

## Usage Notes

- The repository includes token remapping utilities because the project uses a reduced or remapped token vocabulary to keep the model size and training cost manageable. When providing text prompts for generation, the example `generate.py` or notebook demonstrates remapping the tokenizer output before feeding the model and then decoding the model output back to text.
- The example checkpoint and dataset subset are for demonstration — training at larger scale requires more compute and data.

## Reproduce Training

1. Prepare data with `prepare_data.py` to create tokenized `train.bin` and `val.bin` and mapping files.
2. Launch `train.py` with desired hyperparameters and checkpoint/save paths.

Refer to `train.py` for the exact CLI arguments and configurable hyperparameters.

## Two-sample traced training demo

To understand the complete algorithm without loading the multi-gigabyte training set,
run the deterministic two-story teaching demo:

```bash
python trace_train_two_samples.py --steps 20
```

It reuses `GPTLanguageModel` from `model.py`, temporarily selects a CPU-friendly model
size, and prints the in-memory data at each important stage: source text, GPT-2 tokens,
remapped tokens, input/target pairs, embeddings, causal attention, logits, probabilities,
loss, gradients, parameter updates, and autoregressive generation. The regular model
configuration and training files are not modified. The two records live in
`demo_two_stories.jsonl`; edit that file or pass another two-record JSONL file with
`--data`.

## License

This code and model are released under the MIT License. Use and modify freely for research and educational purposes. If you redistribute the trained model or derived works, please include attribution to this repository and retain the MIT license.

## References

- Karpathy’s Zero to Hero Course: YouTube
- nanoGPT GitHub Repo: https://github.com/karpathy/nanoGPT
- TinyStories Dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- Attention is All You Need: https://arxiv.org/abs/1706.03762
- TinyStories Paper: https://arxiv.org/abs/2305.07759
