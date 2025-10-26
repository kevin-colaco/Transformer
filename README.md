# Transformer Architecture from Scratch

A from-scratch implementation of the GPT-2 Transformer architecture in PyTorch, featuring training infrastructure, optimized attention mechanisms, and model deployment capabilities.

## 🚀 Features

- **Pure PyTorch Implementation**: No high-level wrapper dependencies - built from first principles
- **GPT-2 Architecture**: Implements the complete GPT-2 model with configurable sizes
- **Flash Attention**: Uses PyTorch's optimized `scaled_dot_product_attention` for better performance
- **HuggingFace Integration**: Load pretrained GPT-2 weights from HuggingFace Transformers
- **Training Pipeline**: Complete training loop with gradient accumulation, learning rate scheduling, and checkpointing
- **Text Generation**: Autoregressive text generation with top-k sampling
- **Optimized Training**: Fused AdamW, mixed precision (bfloat16), and torch.compile support

## 📁 Project Structure

```
.
├── transformer.py          # Main implementation file
├── checkpoints/           # Training checkpoints
├── input.txt             # Training data
└── README.md
```

## 🏗️ Architecture Components

### Core Modules
- **CausalSelfAttention**: Multi-head self-attention with causal masking
- **MLP**: Gated Linear Unit feed-forward network
- **Block**: Transformer block (attention + MLP with residual connections)
- **GPT**: Complete language model with embedding layers and LM head

### Key Features
- Pre-LayerNorm architecture
- Weight tying (input/output embeddings)
- Gradient checkpointing support
- Configurable model dimensions
- Cosine learning rate decay with warmup

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install torch tiktoken transformers
```

### Basic Usage
```python
from transformer import GPT, GPTConfig

# Initialize model
config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768
)
model = GPT(config)

# Or load pretrained GPT-2
model = GPT.from_pretrained('gpt2')
```

### Training
```bash
python transformer.py
```

The training script includes:
- Gradient accumulation for large batch sizes
- Automatic mixed precision (AMP)
- Model compilation with `torch.compile`
- Checkpoint saving and resumption
- Learning rate scheduling

## ⚙️ Configuration

The model supports all GPT-2 sizes:
- `gpt2` (124M parameters)
- `gpt2-medium` (350M parameters) 
- `gpt2-large` (774M parameters)
- `gpt2-xl` (1558M parameters)

Custom configurations can be created via the `GPTConfig` dataclass.

## 🎯 Training Details

### Optimization
- **Optimizer**: AdamW with decoupled weight decay
- **Learning Rate**: Cosine decay from 6e-4 to 6e-5
- **Batch Size**: 524,288 tokens (simulating GPT-3 scale)
- **Gradient Clipping**: Global norm of 1.0

### Infrastructure
- Automatic CUDA detection and optimization
- Fused AdamW when available
- BF16 mixed precision training
- Gradient accumulation for large effective batches

## 📊 Performance Features

- **Flash Attention**: 2x faster attention computation
- **Torch Compile**: Graph optimization for ~30% speedup
- **Checkpointing**: Resume training from any point
- **Efficient Data Loading**: Minimal overhead data pipeline

## 🔬 Educational Value

This implementation serves as an excellent learning resource for understanding:

- Transformer architecture fundamentals
- GPT-style decoder-only models
- Modern deep learning training techniques
- PyTorch best practices and optimization
- Attention mechanisms and their optimizations

## 📝 Example Output

After training, the model can generate text:
```
Thou art the sun that lights my darkest day,
The gentle breeze that sweeps my cares away...
```

## 🤝 Acknowledgments

- Implementation inspired by [Andrej Karpathy's nanogpt](https://github.com/karpathy/nanogpt)
- Architecture based on the Transformer model from the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), implementing the GPT-2 variant as described in ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- Uses HuggingFace Transformers for pretrained weights

## 📄 License

This project is available for educational and research purposes. Please respect the licenses of any pretrained models used.

---

**Note**: This is an educational implementation. For production use, consider using established libraries like HuggingFace Transformers.