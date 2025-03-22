---

```markdown
# 🧠 BlockLLM vs GaLore: Memory-Efficient LLM Training (CPU-Compatible)

This project benchmarks two memory-efficient optimization techniques — **BlockLLM** and **GaLore** — for training transformer-based language models, with a focus on **CPU-only environments**.

It:
- Trains a HuggingFace-compatible language model (e.g., `distilgpt2`) on a sample dataset
- Applies both **BlockLLM** and **GaLore** optimizers
- Tracks training & validation loss across epochs
- Monitors **CPU memory usage** using `psutil`
- Outputs performance comparison plots

---

## 📦 Installation & Running Instructions (Poetry Setup)

This project uses [Poetry](https://python-poetry.org/) for dependency and environment management.

### ✅ Step-by-Step:

1. **Install Poetry** (if not already installed):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install dependencies**:

   From the root of the project folder (where `pyproject.toml` is located):

   ```bash
   poetry install
   ```

3. **Activate the virtual environment**:

   ```bash
   poetry shell
   ```

4. **Run the training & comparison script**:

   ```bash
   python -m examples.train_wikitext
   ```

---

## 🚀 Features

✅ CPU-only compatible (no CUDA required)  
✅ Compares **BlockLLM** and **GaLore** optimizers  
✅ Tracks training/validation loss over time  
✅ Monitors **CPU RAM usage** using `psutil`  
✅ Automatically generates performance plots and a Markdown summary

---

## 🛠 Configurable Arguments

You can customize training via command-line arguments:

| Argument             | Description                                | Default               |
|----------------------|--------------------------------------------|-----------------------|
| `--model_name`       | HuggingFace model name                     | `distilgpt2`          |
| `--dataset`          | Dataset name                               | `HuggingFaceH4/MATH-500` |
| `--batch_size`       | Training batch size                        | `8`                   |
| `--epochs`           | Number of training epochs                  | `3`                   |
| `--methods`          | Optimizers to compare (`blockllm,galore`)  | `blockllm,galore`     |
| `--output_dir`       | Output folder for plots & logs             | `./results`           |

---

## 📁 Outputs

After training, you'll get:

- 📉 **Loss plot**:  
  `results/cpu_optimization_comparison.png`  
  → Compares training and validation loss for BlockLLM and GaLore

- 📋 **Markdown results table**:  
  `results/comparison_results.md`  
  → Summarizes final losses, average epoch time, and peak CPU memory

---

## 📊 Sample Output Table

| Method   | Final Train Loss | Final Val Loss | Avg Epoch Time | Peak Memory |
|----------|------------------|----------------|----------------|-------------|
| BlockLLM | 3.6753           | 3.5486         | 254.42s        | 1.02 GB     |
| GaLore   | 4.6218           | 4.5899         | 541.84s        | 1.31 GB     |

---

## 🧱 Optimizer Summary

| Optimizer  | Strategy                               | Ideal For               |
|------------|-----------------------------------------|--------------------------|
| BlockLLM   | Sparse updates on block-partitioned params | Memory-constrained training |
| GaLore     | Low-rank approximations of gradients    | Balanced memory-performance tradeoff |

---

## 👏 Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [BlockLLM (RAIVN Lab)](https://github.com/RAIVNLab/blockllm)
- [GaLore (UCSD + CMU)](https://github.com/Liuhong99/GaLore)

---

## 🪪 License

This project is licensed under the **MIT License**.
```

---

Let me know if you'd like to include badges, a demo GIF, or push this to GitHub Pages for visualization!