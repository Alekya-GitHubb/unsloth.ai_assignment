🦥 Unsloth LLM Fine-Tuning & Reinforcement Learning Experiments
This repository demonstrates multiple LLM fine-tuning pipelines using Unsloth — from full fine-tuning and LoRA to reinforcement learning (RLHF & GRPO) and continued pre-training.
All experiments are executed on Google Colab (Pro) using small- to mid-scale open-weights models (Smollm2, Llama 3, Gemma 2, Phi-3, Mistral).

📘 Quick Navigation
#TaskTechniqueColab Link1️⃣Full Fine-TuningTrain all parameters on Smollm2 (135M)Open Colab 1 – Full Fine-Tuning2️⃣LoRA Fine-TuningParameter-efficient adaptation (LoRA)Open Colab 2 – LoRA Parameter-Efficient Fine-Tuning3️⃣Reinforcement LearningPreference-based RLHF setupOpen Colab 3 – Reinforcement Learning4️⃣GRPO Reasoning RLGuided Reinforcement for Prompt OptimizationOpen Colab 4 – RL with GRPO5️⃣Continued PretrainingDomain/language extensionOpen Colab 5 – Continued Pretraining
(Tip: Replace Alekya-GitHubb/unsloth-finetuning with your repo name if different.)

🧩 Colab 1 – Full Fine-Tuning
Model: Smollm2-135M
Method: Full parameter fine-tuning using 4-bit quantized Unsloth modules.


Train all model weights end-to-end.


Define input format, tokenizer, and dataset layout.


Visualize loss curves and validation accuracy.


🔗 Resources:


Unsloth Fine-Tuning Guide


Medium Article – LORA with Ollama



⚙️ Colab 2 – LoRA Parameter-Efficient Fine-Tuning
Model: Smollm2-135M
Method: Low-Rank Adaptation (LoRA).


Freeze base weights, train only LoRA adapters.


Tune parameters: r, alpha, dropout.


8-10× less VRAM than full fine-tuning.


🔗 Resources:


LoRA Docs – Unsloth



🎯 Colab 3 – Reinforcement Learning (RLHF)
Goal: Teach LLMs preferences via human-feedback-style signals.


Use a dataset of preferred vs rejected responses.


Implement reward and policy models.


Apply gradient updates with reward optimization.


🔗 Resources:


Unsloth Reinforcement Learning Guide



🧠 Colab 4 – Reinforcement Learning with GRPO
Goal: Improve reasoning and logical coherence using GRPO.


Train on problem–solution datasets.


Reward chain-of-thought explanations.


Evaluate reasoning depth and clarity.


🔗 Resources:


Train Your Own Reasoning Model – GRPO Tutorial


Unsloth Blog – RL Reasoning



📚 Colab 5 – Continued Pretraining
Goal: Teach LLMs new domains, languages, or styles via unsupervised pretraining.


Extend a checkpoint’s knowledge on new corpus.


Ideal for domain-specific models (e.g. medical, legal, mental health).


Supports multi-lingual adaptation (e.g. English → Telugu).


🔗 Resources:


Unsloth Continued Pretraining Docs


Medium – Fine-Tuning Phi-3 for Mental Health Chatbot



🧬 Model Families Explored
CategoryModels🦙 Meta LlamaLlama 3 (8B), Llama 3.1 (8B)💎 GemmaGemma 2 (2B & 9B)🪶 MistralMistral v0.3 (7B), Mistral NeMo (12B)🧮 PhiPhi-3 Mini & Medium🧠 Tiny ModelsSmollm2 (135M), TinyLlama (1.1B)🐦 QwenQwen2 (7B)

⚙️ Setup
# Clone the repo
git clone https://github.com/Alekya-GitHubb/unsloth-finetuning.git
cd unsloth-finetuning

# Install core dependencies
pip install -U unsloth transformers datasets bitsandbytes accelerate torch


🎥 Video Demonstration Checklist
✅ Explain each notebook’s objective.
✅ Walk through dataset and training cells.
✅ Highlight key metrics (loss, accuracy).
✅ Demonstrate model inference (before vs after tuning).
✅ Summarize results in your own voice.

🔗 Official References


Unsloth Docs


Fine-Tuning Guide


Reinforcement Learning Guide


GRPO Tutorial


Medium – Ollama + LORA



👩‍💻 Author
Alekya Gudise
🎓 MS Software Engineering, San José State University
💼 Ex-LTIMindtree QA Engineer | Python Automation | Cisco Infrastructure
🌐 GitHub @Alekya-GitHubb

Would you like me to generate Colab “Open in Colab” badges (colored buttons) for each notebook instead of plain links?
It’ll make the README even more polished visually.Is this conversation helpful so far?
