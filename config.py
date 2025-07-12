import torch
import random


# global
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_SCALE = 1

IMAGE_MODEL = "sd-legacy/stable-diffusion-v1-5"
CLIP_MODEL = "openai/clip-vit-large-patch14"
LORA_PATH = "lora_weights/Gigachadv1.safetensors"
LORA_ADAPTER_NAME = "Gigachadv1"

ALPHA = 1.5
BETA = 1.0
GAMMA = 1.3


# for step 1
MAX_GENERATIONS = 10
POPULATION_SIZE = 16
MAX_PROMPT_LENGTH = 8
SIMILARITY_TOP_N = 3000
NUM_IMAGES_PER_EVAL = 10
SEEDS = [random.randint(0, 2**32) for _ in range(NUM_IMAGES_PER_EVAL)]

NUM_INFERENCE_STEPS = 25

# for step 2
STEP2_NUM_OPTIM_STEPS         = 150
STEP2_NUM_IMAGES        = 5
STEP2_LR                = 1e-4
STEP2_GUIDANCE_SCALE    = 7.5
STEP2_NUM_INFERENCE_STEPS = 25
STEP2_NUM_IMAGES_PER_PROMPT = 1
STEP2_DO_CLASSIFIER_FREE        = True
STEP2_NEGATIVE_PROMPT           = None