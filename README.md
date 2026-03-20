<div align="center">

# ⚡ Lightweight Fine-Tuning to Foundation Models

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface" />
  <img src="https://img.shields.io/badge/PEFT-LoRA-8B5CF6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  <strong>Parameter-Efficient Fine-Tuning (PEFT) of a pre-trained foundation model using LoRA — adapting powerful LLMs to domain-specific tasks with minimal compute and maximum efficiency.</strong>
</p>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [What is PEFT / LoRA?](#-what-is-peft--lora)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Notebook](#running-the-notebook)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [License](#-license)
- [Author](#-author)

---

## 🧠 Overview

**Lightweight Fine-Tuning to Foundation Models** demonstrates how to efficiently adapt large pre-trained language models (LLMs) to specific downstream tasks using **Parameter-Efficient Fine-Tuning (PEFT)** — specifically the **LoRA (Low-Rank Adaptation)** technique.

Rather than full fine-tuning (which requires updating billions of parameters), this project inserts small, trainable rank-decomposition matrices into a frozen foundation model, reducing trainable parameters by **99%+** while retaining most of the model's capabilities. This makes fine-tuning accessible without enterprise-grade GPU infrastructure.

---

## 🔬 What is PEFT / LoRA?

**PEFT (Parameter-Efficient Fine-Tuning)** is a family of techniques that fine-tune only a small subset of model parameters:

| Method | Description |
|---|---|
| **LoRA** | Injects low-rank matrices into attention layers |
| **Prefix Tuning** | Prepends trainable tokens to input |
| **Prompt Tuning** | Learns soft prompt embeddings |
| **Adapters** | Adds small trainable modules between layers |

**LoRA** (used in this project) approximates weight updates as:

```
W' = W + ΔW = W + B × A
```

Where `B` and `A` are low-rank matrices (rank `r << d`), dramatically reducing trainable parameters while preserving model performance.

---

## ✨ Features

- ✅ **LoRA fine-tuning** with HuggingFace PEFT library
- ✅ **Frozen base model** — only adapter weights are trained
- ✅ **Pre/post evaluation** — compare base vs. fine-tuned model metrics
- ✅ **Minimal VRAM requirement** — runs on consumer GPUs (≥8GB)
- ✅ **Reusable adapter weights** — save, share, and load LoRA adapters independently
- ✅ **Trainer API integration** — clean, reproducible training loop

---

## 🏗️ Architecture

```
Pre-trained Foundation Model (Frozen)
         │
         ├──── Attention Layer ──── LoRA Adapter (A × B, rank r)
         │                               ▲ Only these weights train
         ├──── Attention Layer ──── LoRA Adapter
         │
         └──── FFN Layer ─────────── (Frozen)

Training: minimize task loss using labeled dataset
Inference: merge adapter weights → no latency overhead
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.0 |
| Foundation Model | GPT-2 / DistilBERT / similar HuggingFace LLM |
| Fine-Tuning | HuggingFace `peft` (LoRA) |
| Training | HuggingFace `transformers` Trainer API |
| Tokenization | HuggingFace `tokenizers` |
| Dataset | HuggingFace `datasets` |
| Evaluation | `evaluate` library (accuracy, F1) |
| Notebook | Jupyter |

---

## 🚀 Getting Started

### Prerequisites

- `Python 3.10` or higher
- CUDA-capable GPU (recommended; CPU also supported for small models)
- HuggingFace account (free) for model access

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YeshwanthrajG/LightWeight-FineTuning-to-Foundation-Model.git
cd LightWeight-FineTuning-to-Foundation-Model

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Core dependencies [`requirements.txt`](./requirements.txt):**
```
transformers==4.36.2
datasets==2.16.1
torch==2.0.1
peft==0.7.1
evaluate==0.4.1
scikit-learn==1.3.0
```

### Running the Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: LightWeight_FineTuning.ipynb
# Run all cells sequentially
```

The notebook walks through:
1. Loading the pre-trained foundation model and tokenizer
2. Preparing and tokenizing the dataset
3. Configuring the LoRA adapter (`LoraConfig`)
4. Training with HuggingFace `Trainer`
5. Evaluating base model vs. fine-tuned model
6. Saving and reloading the PEFT adapter

---

## 📊 Results

| Model | Accuracy | Parameters Trained |
|---|---|---|
| Base Foundation Model | ~XX% | 0 (frozen) |
| LoRA Fine-Tuned | ~XX% | < 1% of total |

> Results vary by task and dataset. The fine-tuned adapter achieves significant performance gains while training only a fraction of parameters.

---

## 📁 Project Structure

```
LightWeight-FineTuning-to-Foundation-Model/
├── LoraFineTunedModel   
├── LightWeight_FineTuning.ipynb    # Main project notebook
├── requirements.txt                # Python dependencies
├── saved_model/                    # Trained LoRA adapter weights
├── LICENSE
└── README.md                       # Project documentation
```

---

## 📄 License

This project is licensed under the [`LICENSE`](./LICENSE).

---

## 👤 Author

[@YeshwanthrajG](https://github.com/YeshwanthrajG)
