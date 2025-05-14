# Uncovering Activation Keys in the Dark: Revealing Learned Concepts in LoRA Text-To-Image Models

## Abstract

Low-Rank Adaptation (LoRA) facilitates efficient fine-tuning of text-to-image diffusion models and has driven rapid growth in AI-generated content.
However, its accessibility also raises concerns about misuse and accountability, as malicious actors may distribute LoRA models, either as parameter files or services, that generate inappropriate content while privately sharing information about its usage, such as trigger words or phrases.  
In this paper, we consider an investigator who wants to uncover the purpose of a suspect LoRA. We formalize the investigation goal as uncovering of an activation key, which is a text embedding that triggers generation of a distinct concept in the fine-tuned LoRA model, but not in the base model. The objective function comprises of two components: (1) the intra-model spread, encouraging the chosen key to produce more focused outputs from the fine-tuned model (LoRA) while increasing variability in the base model. (2) The inter-model similarity, penalizing keys that induce similar concepts across both models. We employ a two-stage search strategy. First, an evolutionary search over the text token space identifies a good candidate; then, a refinement via gradient ascent in the text embedding space.
The proposed method is validated on six LoRA models, including stylistic, identity-based, and character concepts demonstrating that our approach reliably uncovers ground-truth concepts.

---

## 🔍 Project Description

This repository contains code and utilities to reproduce the methodology described in our paper:  
**"Uncovering Activation Keys in the Dark: Revealing Learned Concepts in LoRA Text-To-Image Models."**
The core pipeline is divided into:

- **Stage 1: Evolutionary Search**  
  Black-box search over discrete token prompts using NLTK-derived vocabulary and LoRA metadata.  
  Prompts are evolved via mutation and crossover based on a fitness function.
- **Stage 2: Embedding Optimization**  
  White-box refinement via gradient ascent directly in the text embedding space, guided by intra/inter-model CLIP similarities.

## Running the Pipeline

### 1. Evolution + Optimization (Full Pipeline)

```bash
python main.py \
  --sd_model "sd-legacy/stable-diffusion-v1-5" \
  --clip_model "openai/clip-vit-large-patch14" \
  --lora_path "<path-to-lora-model>" \
  --adapter_name "<selected-adapter-name>" \
  --seeds 42 1337 2025
```

#### CLI Parameters

| Argument         | Description                                                                 |
| ---------------- | --------------------------------------------------------------------------- |
| `--sd_model`     | Base Stable Diffusion model repo or path                                    |
| `--clip_model`   | CLIP model (used for embedding & similarity calculations)                   |
| `--lora_path`    | Path to LoRA weights (`.safetensors`)                                       |
| `--adapter_name` | LoRA adapter identifier                                                     |
| `--seeds`        | List of seeds used to initialize generation (for spread/similarity metrics) |

---

## Output

- `Best Prompt`: the highest scoring activation key from evolutionary search
- `Best Score`: final score value of the activation key
- `Image Grids`: 10 images per model showcasing LoRA-specific generation behavior
- `Optimization Curve`: plot of objective value vs optimization steps
- `Side-by-side LoRA/Base comparison`: visually shows the effect of the activation key

## LoRA Models Evaluated

| Model                 | Concept Type      | Link                                               |
| --------------------- | ----------------- | -------------------------------------------------- |
| Chinese Watercolor    | Style             | [CivitAI](https://civitai.com/models/26545)        |
| Cyberpunk Anime Style | Style             | [CivitAI](https://civitai.com/models/128568)       |
| GigaChad              | Identity          | [CivitAI](https://civitai.com/models/18177)        |
| Emma Watson           | Real Face         | [CivitAI](https://civitai.com)                     |
| Scarlett Johansson    | Real Face         | [CivitAI](https://civitai.com)                     |
| Synthetic Identity    | AI-generated Face | [TPDNE](https://this-person-does-not-exist.com/en) |

<!-- ---

## 🔬 Citation

If you use this code or methodology, please cite the following (placeholder until final publication):

```bibtex
@article{activationkeys2025,
  title     = {Uncovering Activation Keys in the Dark: Revealing Learned Concepts in LoRA Text-To-Image Models},
  author    = {Anonymous Authors},
  journal   = {Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

--- -->
