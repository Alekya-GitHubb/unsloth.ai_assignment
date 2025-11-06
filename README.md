🧠 LLM Fine-Tuning, LoRA Adaptation & Reinforcement Learning with 🦥 Unsloth

This repository demonstrates a complete hands-on exploration of fine-tuning open-weights Large Language Models (LLMs) using the Unsloth
 framework — from traditional full fine-tuning to parameter-efficient adaptation (LoRA), reinforcement learning with human feedback (RLHF), reasoning enhancement via GRPO, and continued pretraining for language and domain extension.

Each experiment was implemented, tested, and recorded on Google Colab Pro, following the official Unsloth methodology and linked documentation.


|  #  | Experiment                               | Technique                                                          | Colab Notebook                                                                                                                                                                      |
| :-: | :--------------------------------------- | :----------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1️⃣ | **Full Fine-Tuning**                     | End-to-end parameter optimization on `smollm2-135M`                | [Open Colab 1 – Full Fine-Tuning](https://colab.research.google.com/github/Alekya-GitHubb/unsloth-finetuning/blob/main/Colab1-Full_Finetuning.ipynb)                                |
| 2️⃣ | **LoRA Parameter-Efficient Fine-Tuning** | Lightweight adapter training (Low-Rank Adaptation)                 | [Open Colab 2 – LoRA Fine-Tuning](https://colab.research.google.com/github/Alekya-GitHubb/unsloth-finetuning/blob/main/colab2-lora_parameter-efficient-fine-tuning%20%281%29.ipynb) |
| 3️⃣ | **Reinforcement Learning (RLHF)**        | Preference-based reward learning with chosen vs rejected responses | [Open Colab 3 – Reinforcement Learning](https://colab.research.google.com/github/Alekya-GitHubb/unsloth-finetuning/blob/main/colab3_Reinforcement_learning.ipynb)                   |
| 4️⃣ | **Reasoning RL with GRPO**               | Guided Reinforcement for Prompt Optimization (GRPO)                | [Open Colab 4 – RL with GRPO](https://colab.research.google.com/github/Alekya-GitHubb/unsloth-finetuning/blob/main/colab4-reinformcement%20learning%20with%20grpo.ipynb)            |
| 5️⃣ | **Continued Pretraining**                | Unsupervised domain / language adaptation                          | [Open Colab 5 – Continued Pretraining](https://colab.research.google.com/github/Alekya-GitHubb/unsloth-finetuning/blob/main/colab5-Continued%20pretraining.ipynb)                   |


🧩 Experiment 1 — Full Fine-Tuning

Model: Smollm2-135M
Objective: Perform a complete fine-tune using 4-bit quantized weights (unsloth-bnb-4bit) to update all model parameters.

🔍 Key Aspects

Full gradient update (no frozen layers).

Demonstrates tokenization, data formatting and loss monitoring.

Ideal for understanding end-to-end LLM optimization.

📘 References:

Unsloth Fine-Tuning Guide

Medium – LORA + Ollama Lightweight Solution

⚙️ Experiment 2 — LoRA Parameter-Efficient Fine-Tuning

Objective: Replicate Experiment 1 using LoRA (Low-Rank Adaptation) to achieve parameter efficiency.

🔍 Key Aspects

Base weights frozen → only adapter layers train.

Reduces GPU usage by up to 10× compared to full fine-tuning.

Same dataset as Experiment 1, different parameter update strategy.

📘 References:

LoRA Documentation – Unsloth

🎯 Experiment 3 — Reinforcement Learning (RLHF)

Objective: Teach the model preference alignment using chosen vs rejected responses.

🔍 Key Aspects

Trains a reward model and a policy model.

Demonstrates how feedback improves output quality.

Visualizes reward optimization and policy updates.

📘 References:

Unsloth Reinforcement Learning Guide

🧠 Experiment 4 — Reinforcement Learning with GRPO (Reasoning Enhancement)

Objective: Enhance reasoning and logical consistency using Guided Reinforcement for Prompt Optimization (GRPO).

🔍 Key Aspects

Uses problem-solution pairs for reasoning tasks.

Encourages chain-of-thought explanations.

Trains reward model for “reasoning depth” instead of surface accuracy.

📘 References:

GRPO Tutorial – Train Your Own Reasoning Model

Unsloth Blog – RL Reasoning

📚 Experiment 5 — Continued Pretraining

Objective: Make the LLM learn new language patterns or domain knowledge by continuing unsupervised pretraining.

🔍 Key Aspects

Starts from a fine-tuned checkpoint and extends it.

Ideal for specialized domains (healthcare, finance, education).

Can be used for cross-lingual adaptation (e.g., English → Telugu).

📘 References:

Unsloth Continued Pretraining Docs

Medium – Phi-3 Mental Health Chatbot Fine-Tuning

🧬 Model Families Used
Category	Models
🦙 Meta Llama	Llama 3 (8B), Llama 3.1 (8B)
💎 Gemma	Gemma 2 (2B & 9B)
🪶 Mistral	Mistral v0.3 (7B), Mistral NeMo (12B)
🧮 Phi	Phi-3 Mini & Medium
🌱 Tiny Models	Smollm2 (135M), TinyLlama (1.1B)
🐦 Qwen	Qwen2 (7B)
⚙️ Setup & Execution
# 1️⃣ Clone the repository
git clone https://github.com/Alekya-GitHubb/unsloth-finetuning.git
cd unsloth-finetuning

# 2️⃣ Install dependencies
pip install -U unsloth transformers datasets bitsandbytes accelerate torch

# 3️⃣ Run on Google Colab Pro (T4 / A100)
# Open the desired notebook from the table above.

🎥 Suggested Video Demonstration Flow

When recording the Colab walkthrough (video submission):

Introduce the experiment goal (e.g., “Fine-tuning Smollm2 on chat data”).

Show dataset loading and preprocessing.

Explain training parameters and loss trends.

Demonstrate inference (before vs after fine-tuning).

Summarize outcomes and insights.

🧭 References and Learning Resources

Unsloth Official Docs

Fine-Tuning Guide

Reinforcement Learning Guide

GRPO Reasoning Tutorial

Unsloth Blog on RL Reasoning
