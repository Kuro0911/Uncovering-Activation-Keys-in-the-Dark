import torch
import random



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GENERATIONS = 10
POPULATION_SIZE = 16
MAX_PROMPT_LENGTH = 8
SIMILARITY_TOP_N = 3000
NUM_IMAGES_PER_EVAL = 10
NUM_INFERENCE_STEPS = 25

LORA_SCALE = 1
SEEDS = [random.randint(0, 2**32) for _ in range(NUM_IMAGES_PER_EVAL)]

IMAGE_MODEL = "sd-legacy/stable-diffusion-v1-5"
