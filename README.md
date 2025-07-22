# Uncovering Activation Keys in the Dark: Revealing Learned Concepts in LoRA Text-To-Image Models

## Abstract

Low-Rank Adaptation (LoRA) facilitates efficient fine-tuning of text-to-image diffusion models and has driven rapid growth in AI-generated content.
However, its accessibility also raises concerns about misuse and accountability, as malicious actors may distribute LoRA models, either as parameter files or services, that generate inappropriate content while privately sharing information about its usage, such as trigger words or phrases.  
In this paper, we consider an investigator who wants to uncover the purpose of a suspect LoRA. We formalize the investigation goal as uncovering of an activation key, which is a text embedding that triggers generation of a distinct concept in the fine-tuned LoRA model, but not in the base model. The objective function comprises of two components: (1) the intra-model spread, encouraging the chosen key to produce more focused outputs from the fine-tuned model (LoRA) while increasing variability in the base model. (2) The inter-model similarity, penalizing keys that induce similar concepts across both models. We employ a two-stage search strategy. First, an evolutionary search over the text token space identifies a good candidate; then, a refinement via gradient ascent in the text embedding space.
The proposed method is validated on six LoRA models, including stylistic, identity-based, and character concepts demonstrating that our approach reliably uncovers ground-truth concepts.

---

## Overview & Motivation

Modern text‑to‑image models like Stable Diffusion can be efficiently adapted using LoRA (Low‑Rank Adaptation) to inject new visual concepts. While powerful, this capability raises concerns about accountability: a malicious or proprietary LoRA adapter could hide unwanted or secret behaviors behind obscure trigger phrases. Our pipeline addresses this by answering:

> **What minimal prompt or embedding causes a LoRA model to produce its unique concept, but not the base model?**

We solve this through two complementary phases:

1. **Evolutionary Search (Black‑Box)**

   - Operates purely via API calls to the diffusion model and CLIP scores
   - Evolves human‑readable comma‑separated prompts using genetic operators
   - Prompts are scored to **maximize LoRA consistency** and **minimize overlap** with base model output
   - Produces an **initial discrete prompt** that approximates the true activation key

2. **Gradient Ascent Optimization (White‑Box)**

   - Converts the Stage 1 prompt into continuous CLIP text embeddings
   - Fine‑tunes embedding offsets (excluding the SOS token) via gradient ascent on the same score
   - Differentiates through the diffusion denoiser (with checkpointing) to refine toward an **optimal embedding**

By combining a black‑box search for interpretability with a white‑box refinement for precision, we reliably recover the exact triggers embedded by LoRA adapters.

---

## Repository Structure

```bash
Uncovering-Activation-Keys-in-the-Dark/
├── main.py                           # High‑level orchestration and I/O
├── stage1_evolutionary_search.py     # Implements PromptEvolution class
├── stage2_gradient_ascent_opt.py     # Implements GradientOptimizer class
├── config.py                         # Global hyperparameters & file paths
├── lora_weights/                     # Place your .safetensors LoRA weights here
├── results/                          # Output images, logs, and plots
└── README.md                         # This documentation
```

Each file is documented with in‑line comments to guide you through the core logic.

---

## Installation & Setup

1. **Clone and install dependencies**:

   ```bash
   git clone https://github.com/yourorg/Uncovering-Activation-Keys-in-the-Dark.git
   cd Uncovering-Activation-Keys-in-the-Dark
   pip install -r requirements.txt
   ```

2. **Prepare your LoRA weights**:

   - Copy your `adapter.safetensors` file into `lora_weights/`.
   - In `config.py`, set:

     ```python
     LORA_PATH = "lora_weights/adapter.safetensors"
     LORA_ADAPTER_NAME = "adapter"
     ```

3. **Adjust seeds and device (optional)**:

   - Modify `SEEDS` in `config.py` for deterministic sampling.
   - Ensure `DEVICE = "cuda"` if you have a GPU, or `"cpu"` otherwise.

> **Tip**: Create a Conda or virtualenv environment to avoid dependency conflicts:
>
> ```bash
> python3 -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt
> ```

---

## Usage

After configuring `config.py`, simply run:

```bash
python main.py
```

**What happens under the hood?**

- **Stage 1** (`PromptEvolution`): Evolves a population of prompts over `MAX_GENERATIONS`, saving per‑generation image grids to `results/generation_{i}/`. At the end, a grid `results/step_1_out.png` and `results/results.txt` summarize the best discrete prompt and its fitness.
- **Stage 2** (`GradientOptimizer`, only for SD‑1.5): Refines the Stage 1 prompt embedding for `STEP2_NUM_OPTIM_STEPS` iterations. It outputs an objective curve `results/step2_objective_history.png` and a final side‑by‑side grid `results/step_2_out.png` comparing base vs. LoRA outputs under the optimized key.

All intermediate artifacts and logs are stored under `results/` for easy inspection.

---

## Detailed Code Walkthrough

### `main.py`

- **Model setup**: Loads CLIP and Stable Diffusion (with LoRA adapter) using `setup_clip_model` and `setup_pipeline`.
- **Stage 1**: Instantiates `PromptEvolution`, runs `run_evolution()`, and visualizes the top‑scoring prompt.
- **Stage 2**: If using SD‑1.5, instantiates `GradientOptimizer` and calls `run()` on the discovered prompt.
- **Cleanup**: Frees GPU memory and garbage collects to ensure reproducibility.

### `stage1_evolutionary_search.py`

- **Vocabulary generation**: Loads NLTK Brown corpus and WordNet to build a filtered vocabulary of common nouns/adjectives. Also extracts metadata tags from the LoRA file header.
- **Population initialization**: Random comma‑separated token combos of length ≤ `MAX_PROMPT_LENGTH`.
- **Evolution loop**: For each generation:

  - **Evaluation**: Generates `NUM_IMAGES_PER_EVAL` images under base and LoRA models; computes intra‑model spread and inter‑model similarity via CLIP.
  - **Fitness**: Combines these metrics (`ALPHA`, `BETA`, `GAMMA`) into a tanh‑scaled score.
  - **Genetic operations**: Elite selection, mutation (add/remove/replace tokens), crossover, and diversity boosting.

- **Outputs**: Saves grid images per generation and caches prompt scores to `prompt_evolution_cache.json`.

### `stage2_gradient_ascent_opt.py`

- **Embedding setup**: Encodes Stage 1 prompt to CLIP text embeddings, splits off the SOS token.
- **Optimization**: Treats the remaining embeddings as `torch.nn.Parameter`, uses Adam to maximize the same CLIP‑based objective for `STEP2_NUM_OPTIM_STEPS`. Backpropagates through denoising (via gradient checkpointing) for improved memory efficiency.
- **Outputs**: Exports the optimization curve and a final comparison grid showcasing base vs. LoRA under the refined activation key.

### `config.py`

Centralizes all tunable parameters:

- **Model & adapter paths** (`IMAGE_MODEL`, `CLIP_MODEL`, `LORA_PATH`, `LORA_ADAPTER_NAME`)
- **Search & optimization hyperparameters** (`MAX_GENERATIONS`, `POPULATION_SIZE`, `NUM_INFERENCE_STEPS`, `STEP2_LR`, etc.)
- **Objective weights** (`ALPHA`, `BETA`, `GAMMA`)
- **Reproducibility** (`SEEDS`, `DEVICE`)

Adjust these values to suit your hardware and use case.

---

## Interpreting the Results

- **`results/results.txt`**: Contains the best discrete prompt and its fitness score from Stage 1, plus total runtime.
- **Image grids** (`*.png`): Visual comparisons of base vs. LoRA outputs under each candidate key—inspect for concept specificity.
- **Generation folders** (`results/generation_{i}/`): Track how prompts evolve over time.
- **Objective curves** (`step2_objective_history.png`): Visualize convergence of the gradient ascent stage.

Check these artifacts to verify that the recovered activation key indeed elicits the intended concept only in the fine‑tuned model.

---

## LoRA Models Evaluated

| Model                 | Concept Type      | Link                                               |
| --------------------- | ----------------- | -------------------------------------------------- |
| Chinese Watercolor    | Style             | [CivitAI](https://civitai.com/models/26545)        |
| Cyberpunk Anime Style | Style             | [CivitAI](https://civitai.com/models/128568)       |
| GigaChad              | Identity          | [CivitAI](https://civitai.com/models/18177)        |
| Emma Watson           | Real Face         | [CivitAI](https://civitai.com)                     |
| Scarlett Johansson    | Real Face         | [CivitAI](https://civitai.com)                     |
| Synthetic Identity    | AI-generated Face | [TPDNE](https://this-person-does-not-exist.com/en) |

> ⚠️ **Note**: Due to new government regulations and pressure from Mastercard and Visa, CivitAI has removed many LoRA models especially those involving real identities like celebrities from public access. However, these models can still be downloaded **for testing purposes only** from the following archive: [ALL_MODELS](https://drive.google.com/drive/folders/1gczVTPi9C1AO9X_5lZ670Xrg0V8WHYzp?usp=sharing).

> Read more about this policy change here: [Unite.AI Article](https://www.unite.ai/civitai-tightens-deepfake-rules-under-pressure-from-mastercard-and-visa/)
