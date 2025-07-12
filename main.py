import time
import torch
import gc
import os
import matplotlib.pyplot as plt
from matplotlib.image import imsave

from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline
from stage1_evolutionary_search import PromptEvolution
from stage2_gradient_ascent_opt import GradientOptimizer

from config import DEVICE, SEEDS, IMAGE_MODEL, CLIP_MODEL, LORA_PATH, LORA_ADAPTER_NAME, SEEDS

def setup_clip_model(model_name, device):
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    return clip_model, clip_processor

def setup_pipeline(model_name, lora_path=None, adapter_name=None, device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    if lora_path and adapter_name:
        pipe.load_lora_weights(
            lora_path, weight_name=adapter_name, adapter_name=adapter_name)
        pipe.set_adapters([adapter_name])
    return pipe

def visualize_images(images, best_prompt, best_score, elapsed_time, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, "results.txt")
    with open(txt_path, "w") as f:
        f.write(f"Best Prompt: {best_prompt}\n")
        f.write(f"Best Score: {best_score:.4f}\n")
        f.write(f"Elapsed Time: {elapsed_time}\n")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    title = f"Best Prompt: {best_prompt} | Best Score: {best_score:.4f} | Time: {elapsed_time}"
    fig.suptitle(title, fontsize=12)

    for i in range(min(10, len(images))):
        row, col = divmod(i, 5)
        img = images[i]
        axes[row, col].imshow(img)
        axes[row, col].axis("off")

        img_path = os.path.join(output_dir, f"image_{i}.png")
        imsave(img_path, img)

    montage_path = os.path.join(output_dir, "step_1_out.png")
    fig.savefig(montage_path, bbox_inches="tight")
    plt.close(fig)
           
def main():
    # Load models
    print("-"*45)
    print(f"SEEDS: {SEEDS}")
    print("-"*45)
    
    clip_model, clip_processor = setup_clip_model(CLIP_MODEL, DEVICE)
    base_pipe = setup_pipeline(IMAGE_MODEL, device=DEVICE)
    lora_pipe = setup_pipeline(IMAGE_MODEL, lora_path=LORA_PATH, adapter_name=LORA_ADAPTER_NAME, device=DEVICE)

    # Run evolution
    start_time = time.time()
    evolver = PromptEvolution(base_pipe, lora_pipe, clip_model, clip_processor)
    best_prompt, best_score = evolver.run_evolution()

    print(f"\nOptimal Prompt: {best_prompt}")
    print(f"Best Score: {best_score:.4f}")

    # Generate images
    generated_images = [lora_pipe(best_prompt, num_inference_steps=30).images[0] for _ in range(10)]

    # Visualize
    elapsed_time = time.strftime(
        "%H:%M:%S", time.gmtime(time.time() - start_time))
    visualize_images(generated_images, best_prompt, best_score, elapsed_time)

    # Cleanup
    del lora_pipe, base_pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    # Run next step (if SD-1.5)
    if IMAGE_MODEL == "sd-legacy/stable-diffusion-v1-5":
        optimizer = GradientOptimizer(clip_model)
        optimizer.run(best_prompt)
    
if __name__ == "__main__":
    main()