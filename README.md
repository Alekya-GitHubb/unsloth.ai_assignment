🚀 Unsloth LLM Fine-Tuning & Reinforcement Learning Experiments

This repository showcases a complete series of LLM fine-tuning experiments using the 🦥 Unsloth framework
, covering everything from full fine-tuning to LoRA, reinforcement learning, and continued pretraining.

Each Colab notebook in this repository demonstrates a distinct methodology for adapting and enhancing open-weights language models such as Smollm2, Llama 3, Gemma 2, Phi-3, and Mistral.


🌈 1. Full Fine-Tuning (Colab 1)
Notebook: 👉 Open Colab 1
Model: smollm2-135M
Technique: Full-parameter fine-tuning using unsloth-bnb-4bit adapters.
🔹 Key Points


Uses Unsloth’s full fine-tuning pipeline on a small LLM for demonstration.


Explains input formats, tokenization, and dataset preparation.


Includes video demonstration steps for complete workflow explanation.


Flexible across chat, coding, or Q&A datasets.


🔹 References


📘 Unsloth Fine-Tuning Guide


📖 Medium: LORA + Ollama Lightweight Solution



⚙️ 2. LoRA Parameter-Efficient Fine-Tuning (Colab 2)
Notebook: 👉 Open Colab 2
Model: smollm2-135M
Technique: Low-Rank Adaptation (LoRA) for lightweight fine-tuning.
🔹 Key Points


Converts full fine-tuning into parameter-efficient training.


Freezes base model weights — updates only LoRA adapter matrices.


Reduces GPU memory use by up to 10×.


Configurable parameters: r, alpha, and dropout.


🔹 References


📘 Unsloth LoRA Documentation



🎯 3. Reinforcement Learning (RLHF Setup) (Colab 3)
Notebook: 👉 Open Colab 3
Technique: Supervised + reward-based Reinforcement Learning.
Goal: Teach the model preference alignment using chosen vs rejected examples.
🔹 Key Points


Implements a reward model and policy model setup.


Simulates human feedback loops.


Demonstrates preference scoring and loss optimization.


Visualizes policy updates during reinforcement steps.


🔹 References


📘 Unsloth RL Guide



🧩 4. Reinforcement Learning with GRPO (Colab 4)
Notebook: 👉 Open Colab 4
Technique: GRPO – Guided Reinforcement for Prompt Optimization.
Goal: Enhance reasoning ability using problem-solution datasets.
🔹 Key Points


Uses GRPO for improved logical reasoning in responses.


Incorporates chain-of-thought optimization.


Trains the model to generalize and justify its outputs.


Builds upon reinforcement pipeline with custom reward functions.


🔹 References


📘 GRPO Tutorial


🧩 Unsloth Blog – RL Reasoning



📚 5. Continued Pretraining (Colab 5)
Notebook: 👉 Open Colab 5
Technique: Continued Pretraining / Domain Adaptation.
Goal: Make LLMs learn a new language, style, or domain.
🔹 Key Points


Performs unsupervised continued learning on new corpora.


Extends a model’s knowledge without forgetting previous tasks.


Supports cross-lingual adaptation (e.g., English → Telugu).


Useful for specialized domains (medical, finance, mental health, etc.).


🔹 References


📘 Continued Pretraining Guide


🧠 Medium – Mental Health Chatbot Fine-Tuning Example



🧩 Model Families Used
CategoryModels Explored🦙 Meta LlamaLlama 3 (8B), Llama 3.1 (8B)🪶 MistralMistral v0.3 (7B), Mistral NeMo (12B)💎 GemmaGemma 2 (2B & 9B)🧮 PhiPhi-3 (Mini & Medium)🐦 QwenQwen2 (7B)🌱 Tiny ModelsSmollm2 (135M), TinyLlama (1.1B)

⚙️ Setup Instructions
# Clone the repository
git clone https://github.com/<your-username>/unsloth-finetuning.git
cd unsloth-finetuning

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install core dependencies
pip install -U unsloth transformers datasets bitsandbytes accelerate torch


🎥 Video Walkthrough (Suggested for Submission)
For each notebook:
1️⃣ State the objective (e.g., “Fine-tuning Smollm2 on chat dataset”).
2️⃣ Show key code cells and output logs.
3️⃣ Explain the parameters and datasets used.
4️⃣ Demonstrate inference (before and after fine-tuning).
5️⃣ Summarize results with visual or text metrics.

📊 Suggested Extensions


🧩 Export fine-tuned models to Ollama for local deployment.


🔁 Chain continued pretraining + LoRA for hybrid experiments.


🤖 Integrate Unsloth + LangChain for RAG use cases.


💬 Develop a mental-health chatbot using fine-tuned Phi-3 or Smollm2.



🔗 Useful Resources


Unsloth Docs


Fine-Tuning Guide


Reinforcement Learning Guide


GRPO Tutorial


Medium – LORA with Ollama Lightweight Solution



