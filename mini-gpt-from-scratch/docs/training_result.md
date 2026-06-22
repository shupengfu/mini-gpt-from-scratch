# Training Result

## Experiment Settings

- vocab_size: 24
- batch_size: 16
- seq_len: 16
- d_model: 64
- num_heads: 4
- hidden_dim: 256
- num_layers: 2
- num_steps: 1000

## Loss

The training loss decreased from about 3.35 to about 0.17.

This shows that the MiniGPT model successfully learned character-level patterns from the training text.

## Generated Text

Argmax generation:

gpt predicts the next token one by one. transformer is based on self attention...

Random sampling generation:

gpt predicts the next token one bat transhed by one...
Analysis

The model is able to generate text similar to the training corpus.

Argmax decoding is more stable, while random sampling introduces diversity but also produces more errors.

Since the dataset is very small and repeated many times, the model overfits the training text. However, this is acceptable for this educational project because the goal is to verify the GPT architecture and training pipeline.