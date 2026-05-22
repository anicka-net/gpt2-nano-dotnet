# GPT-2 Nano — .NET Inference

A .NET inference engine for [gpt2-nano](https://github.com/kotlarmilos/gpt2-nano), a GPT-2 model trained from scratch in PyTorch. This demonstrates that Python-trained transformer models can run natively in .NET via TorchSharp — no Python runtime needed.

## What this proves

- **Train in Python, infer in .NET** — load SafeTensors weights into a TorchSharp model
- **Full transformer in C#** — embeddings, multi-head attention (SDPA), MLP, pre-norm residuals
- **Production-grade patterns** — streaming generation, configurable sampling, tensor memory management
- **Safe model loading** — SafeTensors only (no pickle, no code execution, validated before loading)

## Quick start

### 1. Get the model weights

**Option A** — Download from HuggingFace and convert to SafeTensors:
```bash
pip install torch safetensors huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
import torch, os

# Download the PyTorch checkpoint
pt_path = hf_hub_download('kotlarmilos/gpt2-nano', 'checkpoints/final.pt')
cp = torch.load(pt_path, map_location='cpu', weights_only=False)

# Convert to SafeTensors (data-only format, no code execution risk)
os.makedirs('weights', exist_ok=True)
save_file({k: v.float() for k, v in cp['model_state_dict'].items()}, 'weights/model.safetensors')
print('Saved weights/model.safetensors')
"
```

You'll also need the tokenizer files:
```bash
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('data/bpe-tokenizer', exist_ok=True)
for f in ['bpe-tokenizer/vocab.json', 'bpe-tokenizer/merges.json']:
    hf_hub_download('kotlarmilos/gpt2-nano', f, local_dir='data')
print('Tokenizer downloaded')
"
```

**Option B** — Train from scratch (~13h on Apple Silicon):
```bash
git clone https://github.com/kotlarmilos/gpt2-nano
cd gpt2-nano
pip install -r requirements.txt
python -m data.bpe_tokenizer && python -m src.gpt
```

Then export the checkpoint to SafeTensors:
```bash
python -c "
import torch; from safetensors.torch import save_file
cp = torch.load('checkpoints/final.pt', map_location='cpu', weights_only=False)
save_file({k: v.float() for k, v in cp['model_state_dict'].items()}, 'weights/model.safetensors')
"
```

### 2. Run inference

```bash
cd dotnet/Gpt2Nano
dotnet run
```

Or with custom paths and prompt:
```bash
dotnet run -- --model /path/to/model.safetensors \
              --tokenizer /path/to/tokenizer/dir \
              --prompt "The meaning of "
```

### Expected output

```
GPT-2 Nano loaded: 44,062,661 parameters
  Weights: model.safetensors (SafeTensors)
  Attention: scaled_dot_product_attention

» The United States was on March 19, 1915 by the United States General Assembly...
» Science is a technique that involves all of the tasks of the tasks and is not...
» In the year of the Gulf and the USA have been initiated by the US National...
```

The model generates grammatically plausible but factually meaningless text — expected for a 44M parameter model trained on 99M tokens.

## Architecture

```
Tokens (B, C)
  → Embedding (9157, 512) + Sinusoidal Position Encoding
  → 12× Transformer Block:
      Pre-LayerNorm → Multi-Head Attention (8 heads, SDPA) → Residual
      Pre-LayerNorm → MLP (512 → 2048 → 512, ReLU) → Residual
  → Linear Head (512, 9157)
  → Softmax → Sample (temperature / top-k / top-p)
```

## Features

| Feature | Implementation |
|---|---|
| Weight format | SafeTensors with full validation (bounds, overlaps, shape consistency) |
| Attention | `scaled_dot_product_attention` — can dispatch to FlashAttention when available |
| Sampling | Temperature, top-k, top-p (nucleus) — configurable via `SamplingConfig` |
| Streaming | `IAsyncEnumerable<string>` — tokens yielded as generated |
| Memory | `DisposeScope` per generation step — no tensor leaks |
| Tokenizer | Custom BPE (matching training). For standard models, swap to `Microsoft.ML.Tokenizers` |

## Requirements

- .NET 10+
- TorchSharp 0.107.0 (pulled via NuGet)
- `libomp` on macOS: `brew install libomp`

## File layout

```
dotnet/
├── README.md
└── Gpt2Nano/
    ├── Gpt2Nano.csproj    # TorchSharp + TorchSharp-cpu packages
    └── Program.cs          # Everything: model, tokenizer, generation, weight loading
```

The entire implementation is a single `Program.cs` (~440 lines) — intentionally kept in one file so you can read it top to bottom and understand the full inference pipeline.

## Related

- **Training code**: [kotlarmilos/gpt2-nano](https://github.com/kotlarmilos/gpt2-nano)
- **Pre-trained weights**: [kotlarmilos/gpt2-nano on HuggingFace](https://huggingface.co/kotlarmilos/gpt2-nano)
- **Architecture walkthrough**: See `WALKTHROUGH.md` in the training repo for a detailed explanation of every component
