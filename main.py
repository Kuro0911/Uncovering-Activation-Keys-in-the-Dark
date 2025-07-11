import argparse
import time
import torch
import gc
import matplotlib.pyplot as plt

from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline
from Step1 import PromptEvolution
from Step2 import run

from config import DEVICE, SEEDS

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

    montage_path = os.path.join(output_dir, "montage.png")
    fig.savefig(montage_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved 10 images + montage to '{output_dir}/', and summary to '{txt_path}'.")
           
def main(args):
    # Load models
    clip_model, clip_processor = setup_clip_model(args.clip_model, DEVICE)
    base_pipe = setup_pipeline(args.sd_model, device=DEVICE)
    lora_pipe = setup_pipeline(args.sd_model, lora_path=args.lora_path, adapter_name=args.adapter_name, device=DEVICE)

    # Run evolution
    start_time = time.time()
    evolver = PromptEvolution(base_pipe, lora_pipe, clip_model, clip_processor, args.lora_path)
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

    # # Run next step
    # run(args.seeds, best_prompt, args.lora_path, clip_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA Prompt Evolution Runner")

    parser.add_argument("--sd_model", type=str, default="sd-legacy/stable-diffusion-v1-5",
                        help="Path or repo name of Stable Diffusion model")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14", help="Path or repo name of CLIP model")
    parser.add_argument("--lora_path", type=str, default="lora_weights/Gigachadv1.safetensors", help="Path to the LoRA weights")
    parser.add_argument("--adapter_name", type=str,default="Gigachadv1", help="Adapter name for LoRA weights")

    args = parser.parse_args()
    main(args)
