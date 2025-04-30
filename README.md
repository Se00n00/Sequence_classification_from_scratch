# Transformer from Scratch for Emotion Classification

This repository demonstrates a **Transformer-based sequence classification model built from scratch** in PyTorch. The model is fine-tuned on the [`tweet_eval`](https://huggingface.co/datasets/tweet_eval) dataset for **emotion classification**, including labels like *anger, joy, optimism, sadness, fear,* and *love*.

---

## Project Structure
```
.
├── Architectures/                        # Model architectures
│   └── Basic_Sequence_classification.py
├── layers/                               # Custom Transformer layers
│   ├── attention.py
│   ├── embedding.py
│   ├── encoderlayer.py
│   └── feedforward.py
├── best_model.pt                         # Saved PyTorch model
├── fine_tune.ipynb                       # Fine-tuning notebook
├── trainer.ipynb                         # Training script/notebook
├── finetuned-assistant/                 # (Optional) Related outputs or helper modules
├── wandb/                                # Weights & Biases logs (if used)
└── README.md                             # Project description
```
---

## Model Overview

The model `Transformer_For_Sequence_Classification2` is a custom implementation resembling the BERT architecture, composed of:

- **Token Embedding**: Converts token IDs to dense vectors.
- **Positional Encoding**: Adds sequence order information.
- **Transformer Encoder**: Custom multi-head self-attention encoder stack.
- **Dropout Layer**
- **Classification Head**: Maps pooled embedding to 6 emotion classes.

You can find the individual building blocks in the `layers/` directory.

---

## Dataset

- **Dataset**: [`tweet_eval`](https://huggingface.co/datasets/tweet_eval)
- **Task**: Emotion classification
- **Classes**: `anger`, `joy`, `optimism`, `sadness`, `fear`, `love`
- **Source**: Twitter

```python
from datasets import load_dataset
dataset = load_dataset("tweet_eval", "emotion")
```
## Training & Fine-tuning

Use the provided notebooks:

- `fine_tune.ipynb`: Fine-tune the model on the `tweet_eval` dataset.
- `trainer.ipynb`: Contains the training loop, evaluation, and logging.

You can save the model using:

```python
torch.save(model.state_dict(), "best_model.pt")
```
