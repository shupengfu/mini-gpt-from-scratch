# **MiniGPT from Scratch**

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





##### **1. Project Motivation**

Large language models such as GPT, Qwen, LLaMA, and DeepSeek are all based on Transformer architectures.



Before using high-level inference frameworks such as Ollama, vLLM, or SGLang, I wanted to understand what happens inside a GPT-style model.



Therefore, I built a minimal character-level GPT model from scratch with PyTorch.



* This project helps me understand:
* How input token IDs are converted into embeddings
* How position information is added
* How Q, K, V are computed in self-attention
* How multi-head attention splits and merges tensor dimensions
* Why causal mask is needed in autoregressive language modeling
* How Transformer Blocks are stacked
* How the model predicts the next token
* How loss is computed and parameters are updated during training
* How a trained model is saved, loaded, and used for generation





##### **2. Model Architecture**

The overall MiniGPT architecture is:



input\_ids

&#x20;   ↓

Token Embedding

&#x20;   ↓

Position Embedding

&#x20;   ↓

Transformer Blocks

&#x20;   ↓

Final LayerNorm

&#x20;   ↓

LM Head

&#x20;   ↓

logits





Each Transformer Block contains:

Input x

&#x20;   ↓

LayerNorm

&#x20;   ↓

Multi-Head Causal Self-Attention

&#x20;   ↓

Residual Connection

&#x20;   ↓

LayerNorm

&#x20;   ↓

Feed Forward Network

&#x20;   ↓

Residual Connection

&#x20;   ↓

Output



The model is a decoder-only Transformer, similar in structure to GPT-style models.





##### **3. File Structure**

mini-gpt-from-scratch/

├── README.md

├── requirements.txt

├── .gitignore

├── mini\_gpt.py

├── multi\_head\_attention.py

├── transformer\_block.py

├── train\_mini\_gpt.py

├── generate\_mini\_gpt.py

└── docs/

&#x20;   ├── shape\_analysis.md

&#x20;   ├── transformer\_block\_analysis.md

&#x20;   ├── training\_analysis.md

&#x20;   └── training\_result.md



File descriptions:

File						Description

multi\_head\_attention.py		Implements Multi-Head Causal Self-Attention

transformer\_block.py			Implements a Transformer Block with Attention, FFN, LayerNorm, and Residual connections

mini\_gpt.py				Defines the full MiniGPT model

train\_mini\_gpt.py			Trains the MiniGPT model with next-token prediction

generate\_mini\_gpt.py		Loads a trained checkpoint and generates text

docs/shape\_analysis.md		Records tensor shape analysis for attention

docs/training\_result.md		Records training loss and generated text examples





##### **4. Environment**

This project was tested on:

Python 3.x

PyTorch

Windows CPU environment



A GPU is not required for this project because the model is very small.



Install dependencies:

pip install -r requirements.txt



The requirements.txt file contains:

torch





##### **5. How to Train**



Run:

python train\_mini\_gpt.py



The training script will:

* Build a small character-level vocabulary
* Convert text into token IDs
* Construct input-target pairs for next-token prediction
* Train the MiniGPT model
* Print training loss
* Save a checkpoint file



After training, the checkpoint will be saved as:

mini\_gpt\_checkpoint.pt





##### **6. Training Objective**



The training objective is next-token prediction.



Given a sequence of characters, the model learns to predict the next character.



For example:

Text: transformer



A training sample can be:

Input x:  trans

Target y: ransf



The model predicts the next character at every position.



This is the same basic training idea used in GPT-style autoregressive language models.





##### **7. Tensor Shape Flow**



For a batch of input token IDs:

input\_ids shape = \[batch\_size, seq\_len]



After token embedding:

token\_emb shape = \[batch\_size, seq\_len, d\_model]



After position embedding:

x shape = \[batch\_size, seq\_len, d\_model]





Inside Multi-Head Self-Attention:

{

Q, K, V before split:

\[batch\_size, seq\_len, d\_model]



Q, K, V after split:

\[batch\_size, num\_heads, seq\_len, d\_head]



Attention scores:

\[batch\_size, num\_heads, seq\_len, seq\_len]



Attention output:

\[batch\_size, seq\_len, d\_model]

}



After the final LM Head:

logits shape = \[batch\_size, seq\_len, vocab\_size]



This means that for every position in every sequence, the model outputs a probability distribution over the vocabulary.





##### **8. Causal Mask**

The model uses a causal mask to prevent each token from attending to future tokens.



For example, when seq\_len = 4, the causal mask is:

1 0 0 0

1 1 0 0

1 1 1 0

1 1 1 1



This means:

* Token 1 can only attend to token 1
* Token 2 can attend to token 1 and token 2
* Token 3 can attend to token 1, token 2, and token 3
* Token 4 can attend to all previous tokens and itself



This is necessary for autoregressive generation.





##### **9. Training Result**

In my experiment, the training loss decreased from about:

3.35

to about:

0.17



This shows that the model successfully learned character-level patterns from the training text.



Example generated text with argmax decoding:

gpt predicts the next token one by one. transformer is based on self attention...



Example generated text with random sampling:

gpt predicts the next token one bat transhed by one...



Argmax decoding is more stable because it always selects the most likely next character.



Random sampling introduces more diversity, but it can also produce more errors.





##### **10. Checkpoint Saving and Loading**

After training, the model checkpoint is saved with:

torch.save(checkpoint, "mini\_gpt\_checkpoint.pt")



The checkpoint contains:

* Model parameters
* Model configuration
* Character-to-index vocabulary mapping
* Index-to-character vocabulary mapping



During inference, the model is loaded with:

checkpoint = torch.load("mini\_gpt\_checkpoint.pt", map\_location="cpu")

model.load\_state\_dict(checkpoint\["model\_state\_dict"])



This separates the training stage from the generation stage.





##### **11. How to Generate Text**

After training, run:

python generate\_mini\_gpt.py



Then enter a starting character or string, for example:

g



The model will generate text autoregressively.



Generation process:

start\_text

&#x20;   ↓

convert characters to token IDs

&#x20;   ↓

model predicts next token

&#x20;   ↓

append predicted token

&#x20;   ↓

repeat

&#x20;   ↓

decode token IDs back to text





##### **12. What I Learned**



Through this project, I learned:

* How GPT-style models process token IDs
* How token embeddings and position embeddings work
* How Q, K, V are computed in self-attention
* How multi-head attention splits and merges tensor dimensions
* Why causal mask is required in autoregressive models
* How Transformer Blocks are built from attention, FFN, LayerNorm, and residual connections
* How next-token prediction training works
* How CrossEntropyLoss is used for language modeling
* How loss.backward() computes gradients
* How optimizer.step() updates model parameters
* How to save and load PyTorch checkpoints
* How autoregressive text generation works





##### **13. Limitations**



This is a very small educational model.



Current limitations:

* It is character-level, not tokenizer-based
* The training dataset is very small
* The model easily overfits the repeated text
* The generated text is not comparable to real large language models
* It is designed for learning, not production use





##### **14. Future Improvements**



Possible next steps:

Use a larger training corpus

* Increase seq\_len
* Increase d\_model
* Add more Transformer layers
* Add temperature, top-k, and top-p sampling
* Use a real tokenizer instead of character-level tokenization
* Compare the implementation with Hugging Face GPT-style models
* Move from model implementation to LLM serving with Ollama and vLLM





##### **15. Project Summary**



This project helped me move from only using large language models through APIs to understanding the internal structure of GPT-style models.



I implemented a minimal GPT model from scratch, trained it with next-token prediction, observed the loss decrease, saved the model checkpoint, and loaded it for autoregressive generation.



This project builds the foundation for further learning in:

* Transformer models
* PyTorch deep learning
* Large language model training
* LLM inference
* vLLM / SGLang serving
* GPU and CUDA optimization

