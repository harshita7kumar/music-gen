# music-gen

# NSynth-MusicGen: Parameter-Efficient Fine-Tuning (LoRA)

A clean pipeline to fine-tune Meta's MusicGen model using Hugging Face tools and Parameter-Efficient Fine-Tuning (PEFT/LoRA). This project demonstrates how to adapt a large conditional audio generation transformer to synthesize specific instrument tones using Google's NSynth dataset.

## Project Overview
* **Base Model:** Meta's `facebook/musicgen-small` (300M parameters)
* **Dataset:** Google NSynth (subset of instrument notes mapped to text descriptions)
* **Training Method:** LoRA (Low-Rank Adaptation) targeting decoder attention layers (`q_proj`, `v_proj`)
* **Hardware Target:** Optimized to run efficiently on a single standard GPU (e.g., NVIDIA T4 on Google Colab)

## 📁 Repository Structure
* `prepare_data.py`: Downloads the NSynth target dataset, extracts files, and structures raw labels into a unified absolute-path `metadata.jsonl` file.
* `train.py`: Sets up the PyTorch Dataset pipeline, injects the LoRA adapters into the transformer decoder, and executes the training pass.
* `generate.py`: Fuses the trained LoRA weights back onto the base MusicGen model and conditions generation on custom text prompts.

## 🛠️ Hyperparameter Highlights
Based on parameter-efficient tuning best practices, the LoRA adapters use the following layout:
* **Rank (r):** 16
* **Target Modules:** Causal attention projection layers (`q_proj`, `k_proj`, `v_proj`, `out_proj`)
* **Dropout:** 0.05 (to prevent stylistic overfitting on small training samples)
* **Learning Rate:** 3e-4

