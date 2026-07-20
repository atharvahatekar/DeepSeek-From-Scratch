# 🐋 DeepSeek-V3 from Scratch

> A complete, from-scratch PyTorch implementation of the **DeepSeek-V3** architecture — including **Multi-Head Latent Attention (MLA)**, a **Mixture-of-Experts (MoE)** feed-forward network with auxiliary-loss-free load balancing, and **Multi-Token Prediction (MTP)** — built end to end with a full training and inference pipeline.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/Status-Complete-success">
</p>

---

## ✨ Overview

This repository reimplements the core innovations behind DeepSeek-V3 in clean, readable PyTorch — no black-box `transformers` model classes, every component written by hand and wired together into a working GPT-style language model. It is meant as both a **learning resource** and a **runnable research playground**: train it on TinyStories in minutes, or scale it up on FineWeb-Edu.

Each headline component ships with a companion Jupyter notebook that derives the idea from first principles before it lands in the modular codebase.

---

## 🧠 Architecture Highlights

| Component | What it does | File |
|-----------|--------------|------|
| **Multi-Head Latent Attention (MLA)** | Compresses keys/values into a low-rank latent space to shrink the KV cache, with a decoupled RoPE pathway for positional information. | [models/MLA_attention.py](models/MLA_attention.py) |
| **Mixture-of-Experts (MoE)** | Top-*k* token routing across expert SwiGLU MLPs plus an always-on shared expert. Uses **auxiliary-loss-free load balancing** via a dynamic per-expert routing bias. | [models/moe.py](models/moe.py) |
| **Multi-Token Prediction (MTP)** | Extra prediction heads that forecast several future tokens per step, densifying the training signal. | [models/mtp.py](models/mtp.py) |
| **RMSNorm · RoPE · SwiGLU** | Hand-written normalization, rotary position embeddings, and gated activations. | [models/layers.py](models/layers.py) |
| **DeepSeekV3 model** | Pre-norm transformer blocks (MLA + MoE), weight-tied LM head, and combined main + MTP loss. | [models/model.py](models/model.py) |

### Why these matter

- **MLA** dramatically reduces memory during inference by caching a compact latent representation instead of full-size keys and values.
- **MoE** scales model capacity without a proportional increase in compute — only a few experts fire per token — while the auxiliary-loss-free scheme keeps experts balanced *without* a competing loss term.
- **MTP** improves sample efficiency and gives the model a richer learning signal by predicting multiple tokens ahead.

---

## 📂 Project Structure

```
DeepSeek-From-Scratch/
├── models/
│   ├── config.py            # DeepSeekConfig dataclass (all hyperparameters)
│   ├── layers.py            # RMSNorm, RotaryEmbedding, RoPE, SwiGLU
│   ├── MLA_attention.py     # Multi-Head Latent Attention
│   ├── moe.py               # Mixture-of-Experts layer
│   ├── mtp.py               # Multi-Token Prediction heads
│   └── model.py             # Full DeepSeekV3 model + generation
├── training/
│   ├── data_loader.py       # Memmapped batching + loss estimation
│   └── trainer.py           # Training loop (AMP, cosine LR, W&B logging)
├── inference/
│   └── generator.py         # Load checkpoint and generate text
├── notebooks/               # From-scratch derivations of each component
│   ├── Multi_Head_Latent_Attention_From_Scratch.ipynb
│   ├── MoE_from_Scratch.ipynb
│   └── Multi_Token_Prediction_from_Scratch.ipynb
├── prepare_data_tiny_stories.py   # Tokenize TinyStories → train/val .bin
├── prepare_data_fineweb.py        # Memory-efficient FineWeb-Edu prep
├── main.py                        # Entry point: demo / train
├── run_inference.py               # Standalone text generation script
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/<your-username>/DeepSeek-From-Scratch.git
cd DeepSeek-From-Scratch
pip install -r requirements.txt
```

### 2. Quick sanity check

Run the built-in demo — it builds a small model, runs a forward pass, reports the main/MTP losses, and generates a few tokens:

```bash
python main.py demo
```

### 3. Prepare data

Tokenize a dataset into `train.bin` / `validation.bin` (GPT-2 BPE via `tiktoken`):

```bash
# Small & fast — great for a first run
python prepare_data_tiny_stories.py

# Larger, higher-quality web text
python prepare_data_fineweb.py
```

### 4. Train

```bash
python main.py train
```

Training features baked in:
- ⚡ Mixed precision (`bfloat16`/`float16`) with `GradScaler`
- 📉 Warmup + cosine learning-rate decay
- 🔁 Gradient accumulation
- 💾 Best-checkpoint saving (`best_deepseek_v3.pt`)
- 📊 [Weights & Biases](https://wandb.ai) logging of every loss component

> **Tip:** Add a `.env` file with your `WANDB_API_KEY`, or run `wandb offline` to log locally.

### 5. Generate text

Edit the prompts at the bottom of [run_inference.py](run_inference.py) and run:

```bash
python run_inference.py
```

```python
generated = model.generate(context, max_new_tokens=80, temperature=0.8, top_k=50)
```

---

## ⚙️ Configuration

All architecture and training knobs live in the [`DeepSeekConfig`](models/config.py) dataclass:

| Group | Parameters |
|-------|-----------|
| **Model** | `vocab_size`, `block_size`, `n_layer`, `n_embd`, `n_head` |
| **MLA** | `kv_lora_rank`, `q_lora_rank`, `rope_dim` |
| **MoE** | `n_experts`, `n_experts_per_token`, `expert_intermediate_size`, `shared_expert_intermediate_size`, `use_shared_expert` |
| **MTP** | `mtp_num_heads`, `mtp_loss_weight` |
| **Regularization** | `dropout`, `bias`, `aux_loss_weight` |

Scale the model up or down by editing the config in [training/trainer.py](training/trainer.py).

---

## 📓 Notebooks

The [`notebooks/`](notebooks/) folder walks through each core idea step by step, from the math to a minimal working implementation — the perfect place to start if you want to *understand* DeepSeek-V3, not just run it:

- **Multi-Head Latent Attention** — low-rank KV compression + decoupled RoPE
- **Mixture-of-Experts** — routing, top-*k* selection, and load balancing
- **Multi-Token Prediction** — predicting multiple future tokens per position

---

## 🗺️ Roadmap / Ideas

- [ ] KV-cache reuse in `generate()` for faster inference
- [ ] FlashAttention / `scaled_dot_product_attention` backend
- [ ] Configurable MTP depth at inference (speculative decoding)
- [ ] Multi-GPU / distributed training

---

## 🙏 Acknowledgements

Inspired by the [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) and the broader nanoGPT-style tradition of learning by rebuilding. Built for education and experimentation.

---

## 📜 License

Released under the MIT License. Feel free to learn from, fork, and build on it.

---

<p align="center"><i>Built from scratch — one attention head, one expert, and one token at a time. 🐋</i></p>
