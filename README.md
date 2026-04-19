# 🧠 Therapy LLM — Fine-tuning LLaMA 3.1-8B for Mental Health Support

A full end-to-end LLM fine-tuning pipeline that adapts **Meta LLaMA 3.1-8B** into a mental-health-aware conversational assistant using **SFT → DPO → RLAIF** training stages on Google Colab (T4 GPU).

---

## 🌟 Project Highlights

| Item | Detail |
|---|---|
| **Base Model** | `meta-llama/Meta-Llama-3.1-8B` (4-bit quantized via Unsloth) |
| **Fine-tuning Method** | LoRA (rank=16, alpha=16) via PEFT |
| **Training Pipeline** | SFT → DPO → RLAIF |
| **Training Platform** | Google Colab (T4 GPU) |
| **Adapter Size** | ~160 MB (LoRA adapter only) |
| **Trained Model** | 🤗 [therapy-teacher-lora on HuggingFace]([SherazTheG/therapy-teacher-lora](https://huggingface.co/SherazTheG/therapy-teacher-lora)) |

---

## 📊 Evaluation Results

| Metric | Base Model | Fine-tuned Model |
|---|---|---|
| Tokens/sec (TPS) | 0.15 | **13.65** |
| VRAM Usage | 5.91 GB | 6.12 GB |
| Empathy Response | ❌ Empty — failed to generate | ✅ Warm, validating response |
| Safety Pass | ❌ No | ✅ Yes |
| Notes | Model gave empty empathy response | Correctly redirected medication request with alternatives |

---

## 📁 Project Structure

```
LLM finetuning/
│
├── 📓 data preprocessing.ipynb     # Local data cleaning & dataset construction
├── 📓 model finetuning.ipynb       # Local finetuning attempt (RTX 5060 Ti)
├── 📓 train (1).ipynb              # Colab: SFT + DPO training (T4 GPU)
├── 📓 rlaif (6).ipynb              # Colab: RLAIF training
├── 📓 eval (1).ipynb               # Evaluation notebook
│
├── 📄 dpo_dataset (2).jsonl        # Processed DPO preference dataset
├── 📄 response_pairs.jsonl         # Response pairs used in RLAIF
├── 📄 evaluation_results.json      # Quantitative eval results
│
└── therapy_teacher_lora/
    ├── adapter_config.json         # LoRA adapter configuration
    ├── tokenizer_config.json       # Tokenizer settings
    └── README.md                   # HuggingFace model card
```

---

## 🛠️ Training Pipeline

### Stage 1 — Data Preprocessing (`data preprocessing.ipynb`)
- Combined **CounselChat** and **Psychology-10K** datasets
- Cleaned, deduplicated, and reformatted into instruction-following format
- Generated `response_pairs.jsonl` for preference learning

### Stage 2 — Supervised Fine-Tuning / SFT (`train (1).ipynb`)
- Fine-tuned LLaMA 3.1-8B using **Unsloth** on Google Colab T4
- Applied **4-bit quantization** (BnB) to fit within 16GB VRAM
- LoRA config: rank=16, alpha=16, targeting all attention + MLP projection layers

### Stage 3 — DPO Training (`train (1).ipynb` / `rlaif (6).ipynb`)
- Ran **Direct Preference Optimization (DPO)** using `dpo_dataset.jsonl`
- Preference pairs selected to reinforce empathetic, safe, and helpful responses

### Stage 4 — RLAIF (`rlaif (6).ipynb`)
- Applied **Reinforcement Learning from AI Feedback** using an LLM-as-judge
- Further aligned responses for safety and emotional sensitivity

### Stage 5 — Evaluation (`eval (1).ipynb`)
- Benchmarked base vs. fine-tuned model on TPS, VRAM, empathy quality, and safety

---

## ⚠️ Hardware Note

Initial fine-tuning was attempted locally on an **NVIDIA RTX 5060 Ti**, but the GPU's CUDA architecture is not fully supported by current PyTorch/Unsloth builds. Training was subsequently migrated to **Google Colab with a T4 GPU**, which provided stable CUDA 12.x support.

---

## 🚀 How to Use the Fine-tuned Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "unsloth/meta-llama-3.1-8b-bnb-4bit"
adapter = "YOUR_HF_USERNAME/therapy-teacher-lora"  # Update with your HuggingFace repo

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, load_in_4bit=True)
model = PeftModel.from_pretrained(model, adapter)

prompt = "I've been feeling really anxious and overwhelmed lately. I don't know what to do."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 📦 Datasets Used

- [CounselChat](https://huggingface.co/datasets/nbertagnolli/counsel-chat) — Real therapy Q&A sessions
- [Psychology-10K](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) — Mental health counseling conversations

---

## 🔧 Key Libraries

| Library | Purpose |
|---|---|
| [Unsloth](https://github.com/unslothai/unsloth) | Fast, memory-efficient LoRA fine-tuning |
| [PEFT](https://github.com/huggingface/peft) | LoRA adapter management |
| [TRL](https://github.com/huggingface/trl) | SFT / DPO / RLAIF trainers |
| [Transformers](https://github.com/huggingface/transformers) | Model loading & inference |
| [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit quantization |

---

## 📋 Disclaimer

This model is a research project and **not a substitute for professional mental health care**. It should not be used for clinical diagnosis or therapy. Always consult a licensed mental health professional.

---

## 👤 Author

Built as a personal ML project to explore LLM fine-tuning for empathetic AI applications.
