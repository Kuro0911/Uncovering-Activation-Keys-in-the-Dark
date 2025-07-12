import torch
import random

# global
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"               # compute device for model inference and training
LORA_SCALE = 1                                                       # scaling factor for LoRA cross‐attention weights

IMAGE_MODEL = "sd-legacy/stable-diffusion-v1-5"                       # base Stable Diffusion model identifier
CLIP_MODEL = "openai/clip-vit-large-patch14"                          # CLIP model identifier for scoring/image features
LORA_PATH = "lora_weights/Gigachadv1.safetensors"                     # path to the fine‐tuned LoRA weights file
LORA_ADAPTER_NAME = "Gigachadv1"                                      # adapter name used when loading LoRA weights

ALPHA = 1.5                                                           # weight for LoRA image diversity penalty in objective
BETA = 1.0                                                            # weight for inter-model similarity term in objective
GAMMA = 1.3                                                           # weight for base image diversity reward in objective

# for step 1 (evolutionary search)
MAX_GENERATIONS = 10                                                  # number of evolutionary generations to run
POPULATION_SIZE = 16                                                  # number of prompts in each generation
MAX_PROMPT_LENGTH = 8                                                 # maximum number of tokens per prompt
SIMILARITY_TOP_N = 3000                                               # number of nearest neighbors when exploring similar words
NUM_IMAGES_PER_EVAL = 10                                               # number of images generated per prompt evaluation
SEEDS = [random.randint(0, 2**32) for _ in range(NUM_IMAGES_PER_EVAL)]  # random seeds for reproducible image sampling

NUM_INFERENCE_STEPS = 25                                               # number of diffusion timesteps for Stage 1

# for step 2 (gradient ascent optimization)
STEP2_NUM_OPTIM_STEPS = 150                                           # number of gradient‐based optimization iterations
STEP2_NUM_IMAGES = 5                                                  # number of images to generate per iteration
STEP2_LR = 1e-4                                                       # learning rate for embedding optimizers
STEP2_GUIDANCE_SCALE = 7.5                                            # classifier-free guidance strength for Stage 2
STEP2_NUM_INFERENCE_STEPS = 25                                        # diffusion timesteps for Stage 2 denoising
STEP2_NUM_IMAGES_PER_PROMPT = 1                                        # images per prompt encoding call
STEP2_DO_CLASSIFIER_FREE = True                                       # whether to use classifier-free guidance
STEP2_NEGATIVE_PROMPT = None                                          # negative prompt text to suppress unwanted content