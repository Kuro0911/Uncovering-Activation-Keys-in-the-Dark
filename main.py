import argparse
import time
import torch
import gc
import matplotlib.pyplot as plt

from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline
from Step1 import PromptEvolution
from Step2 import run


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


def visualize_images(images, best_prompt, best_score, elapsed_time):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(
        f"Best Prompt: {best_prompt} | Best Score: {best_score:.4f} | Time: {elapsed_time}", fontsize=12)

    for i in range(10):
        row, col = divmod(i, 5)
        axes[row, col].imshow(images[i])
        axes[row, col].axis("off")
    plt.show()


def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    clip_model, clip_processor = setup_clip_model(args.clip_model, DEVICE)
    base_pipe = setup_pipeline(args.sd_model, device=DEVICE)
    lora_pipe = setup_pipeline(
        args.sd_model, lora_path=args.lora_path, adapter_name=args.adapter_name, device=DEVICE)

    # Run evolution
    start_time = time.time()
    evolver = PromptEvolution(base_pipe, lora_pipe, args.lora_path)
    best_prompt, best_score = evolver.run_evolution()

    print(f"\nOptimal Prompt: {best_prompt}")
    print(f"Best Score: {best_score:.4f}")

    # Generate images
    generated_images = [
        lora_pipe(best_prompt, num_inference_steps=30).images[0] for _ in range(10)]

    # Visualize
    elapsed_time = time.strftime(
        "%H:%M:%S", time.gmtime(time.time() - start_time))
    visualize_images(generated_images, best_prompt, best_score, elapsed_time)

    # Cleanup
    del lora_pipe, base_pipe
    gc.collect()
    torch.cuda.empty_cache()

    # Run next step
    run(args.seeds, best_prompt, args.lora_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA Prompt Evolution Runner")

    parser.add_argument("--sd_model", type=str, default="sd-legacy/stable-diffusion-v1-5",
                        help="Path or repo name of Stable Diffusion model")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14",
                        help="Path or repo name of CLIP model")
    parser.add_argument("--lora_path", type=str,
                        required=True, help="Path to the LoRA weights")
    parser.add_argument("--adapter_name", type=str,
                        required=True, help="Adapter name for LoRA weights")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2025],
                        help="List of seeds to pass to Step 2")

    args = parser.parse_args()
    main(args)
