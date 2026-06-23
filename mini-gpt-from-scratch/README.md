# MiniGPT from Scratch

This project implements a small character-level GPT model from scratch using PyTorch.

The goal of this project is not to build a large language model, but to understand the core components and training process of GPT-style autoregressive language models.

Through this project, I implemented:

* Token Embedding
* Position Embedding
* Multi-Head Causal Self-Attention
* Transformer Block
* Feed Forward Network
* LayerNorm
* LM Head
* Next-token prediction training
* Checkpoint saving and loading
* Autoregressive text generation

---

## 1. Project Motivation

Large language models such as GPT, Qwen, LLaMA, and DeepSeek are all based on Transformer architectures.

Before using high-level inference frameworks such as Ollama, vLLM, or SGLang, I wanted to understand what happens inside a GPT-style model.

Therefore, I built a minimal character-level GPT model from scratch with PyTorch.

This project helps me understand:

* How input token IDs are converted into embeddings
* How position information is added
* How Q, K, V are computed in self-attention
* How multi-head attention splits and merges tensor dimensions
* Why causal mask is needed in autoregressive language modeling
* How Transformer Blocks are stacked
* How the model predicts the next token
* How loss is computed and parameters are updated during training
* How a trained model is saved, loaded, and used for generation

---

## 2. Model Architecture

The overall MiniGPT architecture is:

```text
input_ids
    ↓
Token Embedding
    ↓
Position Embedding
    ↓
Transformer Blocks
    ↓
Final LayerNorm
    ↓
LM Head
    ↓
logits
```

Each Transformer Block contains:

```text
Input x
    ↓
LayerNorm
    ↓
Multi-Head Causal Self-Attention
    ↓
Residual Connection
    ↓
LayerNorm
    ↓
Feed Forward Network
    ↓
Residual Connection
    ↓
Output
```

The model is a decoder-only Transformer, similar in structure to GPT-style models.

---

## 3. File Structure

```text
mini-gpt-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── mini_gpt.py
├── multi_head_attention.py
├── transformer_block.py
├── train_mini_gpt.py
├── generate_mini_gpt.py
└── docs/
    ├── shape_analysis.md
    ├── transformer_block_analysis.md
    ├── training_analysis.md
    └── training_result.md
```

File descriptions:

| File                      | Description                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------- |
| `multi_head_attention.py` | Implements Multi-Head Causal Self-Attention                                             |
| `transformer_block.py`    | Implements a Transformer Block with Attention, FFN, LayerNorm, and Residual connections |
| `mini_gpt.py`             | Defines the full MiniGPT model                                                          |
| `train_mini_gpt.py`       | Trains the MiniGPT model with next-token prediction                                     |
| `generate_mini_gpt.py`    | Loads a trained checkpoint and generates text                                           |
| `docs/shape_analysis.md`  | Records tensor shape analysis for attention                                             |
| `docs/training_result.md` | Records training loss and generated text examples                                       |

---

## 4. Environment

This project was tested on:

```text
Python 3.x
PyTorch
Windows CPU environment
```

A GPU is not required for this project because the model is very small.

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
torch
```

---

## 5. How to Train

Run:

```bash
python train_mini_gpt.py
```

The training script will:

1. Build a small character-level vocabulary
2. Convert text into token IDs
3. Construct input-target pairs for next-token prediction
4. Train the MiniGPT model
5. Print training loss
6. Save a checkpoint file

After training, the checkpoint will be saved as:

```text
mini_gpt_checkpoint.pt
```

---

## 6. Training Objective

The training objective is next-token prediction.

Given a sequence of characters, the model learns to predict the next character.

For example:

```text
Text: transformer
```

A training sample can be:

```text
Input x:  trans
Target y: ransf
```

The model predicts the next character at every position.

This is the same basic training idea used in GPT-style autoregressive language models.

---

## 7. Tensor Shape Flow

For a batch of input token IDs:

```text
input_ids shape = [batch_size, seq_len]
```

After token embedding:

```text
token_emb shape = [batch_size, seq_len, d_model]
```

After position embedding:

```text
x shape = [batch_size, seq_len, d_model]
```

Inside Multi-Head Self-Attention:

```text
Q, K, V before split:
[batch_size, seq_len, d_model]

Q, K, V after split:
[batch_size, num_heads, seq_len, d_head]

Attention scores:
[batch_size, num_heads, seq_len, seq_len]

Attention output:
[batch_size, seq_len, d_model]
```

After the final LM Head:

```text
logits shape = [batch_size, seq_len, vocab_size]
```

This means that for every position in every sequence, the model outputs a probability distribution over the vocabulary.

---

## 8. Causal Mask

The model uses a causal mask to prevent each token from attending to future tokens.

For example, when `seq_len = 4`, the causal mask is:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

This means:

* Token 1 can only attend to token 1
* Token 2 can attend to token 1 and token 2
* Token 3 can attend to token 1, token 2, and token 3
* Token 4 can attend to all previous tokens and itself

This is necessary for autoregressive generation.

---

## 9. Training Result

In my experiment, the training loss decreased from about:

```text
3.35
```

to about:

```text
0.17
```

This shows that the model successfully learned character-level patterns from the training text.

Example generated text with argmax decoding:

```text
gpt predicts the next token one by one. transformer is based on self attention...
```

Example generated text with random sampling:

```text
gpt predicts the next token one bat transhed by one...
```

Argmax decoding is more stable because it always selects the most likely next character.

Random sampling introduces more diversity, but it can also produce more errors.

---

## 10. Checkpoint Saving and Loading

After training, the model checkpoint is saved with:

```python
torch.save(checkpoint, "mini_gpt_checkpoint.pt")
```

The checkpoint contains:

* Model parameters
* Model configuration
* Character-to-index vocabulary mapping
* Index-to-character vocabulary mapping

During inference, the model is loaded with:

```python
checkpoint = torch.load("mini_gpt_checkpoint.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
```

This separates the training stage from the generation stage.

---

## 11. How to Generate Text

After training, run:

```bash
python generate_mini_gpt.py
```

Then enter a starting character or string, for example:

```text
g
```

The model will generate text autoregressively.

Generation process:

```text
start_text
    ↓
convert characters to token IDs
    ↓
model predicts next token
    ↓
append predicted token
    ↓
repeat
    ↓
decode token IDs back to text
```

---

## 12. What I Learned

Through this project, I learned:

1. How GPT-style models process token IDs
2. How token embeddings and position embeddings work
3. How Q, K, V are computed in self-attention
4. How multi-head attention splits and merges tensor dimensions
5. Why causal mask is required in autoregressive models
6. How Transformer Blocks are built from attention, FFN, LayerNorm, and residual connections
7. How next-token prediction training works
8. How CrossEntropyLoss is used for language modeling
9. How `loss.backward()` computes gradients
10. How `optimizer.step()` updates model parameters
11. How to save and load PyTorch checkpoints
12. How autoregressive text generation works

---

## 13. Limitations

This is a very small educational model.

Current limitations:

* It is character-level, not tokenizer-based
* The training dataset is very small
* The model easily overfits the repeated text
* The generated text is not comparable to real large language models
* It is designed for learning, not production use

---

## 14. Future Improvements

Possible next steps:

* Use a larger training corpus
* Increase `seq_len`
* Increase `d_model`
* Add more Transformer layers
* Add temperature, top-k, and top-p sampling
* Use a real tokenizer instead of character-level tokenization
* Compare the implementation with Hugging Face GPT-style models
* Move from model implementation to LLM serving with Ollama and vLLM

---

## 15. Project Summary

This project helped me move from only using large language models through APIs to understanding the internal structure of GPT-style models.

I implemented a minimal GPT model from scratch, trained it with next-token prediction, observed the loss decrease, saved the model checkpoint, and loaded it for autoregressive generation.

This project builds the foundation for further learning in:

* Transformer models
* PyTorch deep learning
* Large language model training
* LLM inference
* vLLM / SGLang serving
* GPU and CUDA optimization
