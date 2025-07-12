import os

from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import torch
import torchvision
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm

from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

from config import (
  DEVICE,
  SEEDS,
  LORA_PATH,
  STEP2_NUM_OPTIM_STEPS,
  STEP2_NUM_IMAGES,
  STEP2_LR,
  STEP2_GUIDANCE_SCALE,
  STEP2_NUM_INFERENCE_STEPS,
  STEP2_NUM_IMAGES_PER_PROMPT,
  STEP2_DO_CLASSIFIER_FREE,
  STEP2_NEGATIVE_PROMPT,
  ALPHA,
  BETA,
  GAMMA
)

class GradientOptimizer:
    def __init__(self, clip_model):
    
        self.device          = DEVICE
        self.lora_path       = LORA_PATH
        self.num_optimization_steps = STEP2_NUM_OPTIM_STEPS
        self.num_images      = STEP2_NUM_IMAGES
        self.learning_rate   = STEP2_LR
        self.guidance_scale  = STEP2_GUIDANCE_SCALE
        self.num_inference_steps   = STEP2_NUM_INFERENCE_STEPS
    
        self.num_images_per_prompt    = STEP2_NUM_IMAGES_PER_PROMPT
        self.do_classifier_free_guidance       = STEP2_DO_CLASSIFIER_FREE
        self.negative_prompt          = STEP2_NEGATIVE_PROMPT

        self.pipeline_base = self.init_pipeline_base()
        self.pipeline_lora = self.init_pipeline_lora()

        self.clip_model = clip_model

        # Prepare timesteps
        self.scheduler_base = self.pipeline_base.scheduler
        self.scheduler_lora = self.pipeline_lora.scheduler
        
    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        r"""
        Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
        Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
        Flawed](https://arxiv.org/pdf/2305.08891.pdf).
        """
        std_text = noise_pred_text.std(
            dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(
            dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + \
            (1 - guidance_rescale) * noise_cfg
        return noise_cfg

    def init_pipeline_base(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-v1-5")
        pipeline.to(self.device)

        # Freeze the U-Net, VAE, and text encoder, but still compute gradients
        pipeline.unet.requires_grad_(False)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)

        pipeline.unet.enable_gradient_checkpointing()

        return pipeline

    def init_pipeline_lora(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            "sd-legacy/stable-diffusion-v1-5")
        pipeline.load_lora_weights(self.lora_path)
        pipeline.to(self.device)

        # Freeze the U-Net, VAE, and text encoder, but still compute gradients
        pipeline.unet.requires_grad_(False)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)

        pipeline.unet.enable_gradient_checkpointing()

        return pipeline

    def get_text_embedding(self, condition, uncondition, SOS_token):
        cond = torch.cat((SOS_token.unsqueeze(dim=1), condition), dim=1)
        uncond = torch.cat((SOS_token.unsqueeze(dim=1), uncondition), dim=1)
        prompt_embeds = torch.cat([uncond, cond])

        return prompt_embeds

    # Updated version to correctly process the image tensor before computing embedding
    def process_img(self, image_tensor):
        # Convert uint8 images to float and scale to [0, 1]
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0
        else:
            image_tensor = image_tensor.float()

        # Add batch dimension if needed
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
            needs_squeeze = True
        else:
            needs_squeeze = False

        # Get original dimensions
        _, _, h, w = image_tensor.shape

        # Calculate new dimensions maintaining aspect ratio
        if h < w:
            new_h = 224
            new_w = int(w * (224 / h))
        else:
            new_w = 224
            new_h = int(h * (224 / w))

        # Resize with bicubic interpolation
        resized = torch.nn.functional.interpolate(
            image_tensor,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False,
            antialias=True  # Remove if using PyTorch <1.11
        )

        # Center crop to 224x224
        top = (new_h - 224) // 2
        left = (new_w - 224) // 2
        cropped = resized[:, :, top:top+224, left:left+224]

        # CLIP-specific normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=image_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=image_tensor.device).view(1, 3, 1, 1)
        normalized = (cropped - mean) / std

        # Remove batch dimension if added
        if needs_squeeze:
            normalized = normalized.squeeze(0)

        return normalized

    # Denoising loop
    def do_denoise(self, pipe, scheduler, timesteps, latents, prompt_embeds, guidance_scale):
        for i, t in enumerate(tqdm(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(
                latent_model_input, t)

            # Use gradient checkpointing for the unet forward pass
            noise_pred_uncond, noise_pred_text = checkpoint(
                lambda latent_model_input, t, prompt_embeds: pipe.unet(
                    latent_model_input, t, encoder_hidden_states=prompt_embeds).sample.chunk(2),
                latent_model_input, t, prompt_embeds,
                use_reentrant=False  # Explicitly set use_reentrant
            )
            noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_pred_text - noise_pred_uncond)
            noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    # Decode from latent to image tensor
    def latent2img_tensor(self, latent, pipeline):
        # image_tensor = pipeline.vae.decode(latent / pipeline.vae.config.scaling_factor, return_dict=False, generator=generator)[0]

        # Use gradient checkpointing for VAE decode
        scaled_latent = latent / pipeline.vae.config.scaling_factor
        image_tensor = torch.utils.checkpoint.checkpoint(
            lambda ls, v: v.decode(ls, return_dict=False)[
                0],  # Function and arguments
            scaled_latent,  # Scaled latent input
            pipeline.vae,    # VAE model
            use_reentrant=False
        )
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)  # Scale to [0, 1]
        return image_tensor

    def compute_clip_embeddings(self, image_tensors):
        clip_embeddings = []
        for tensor in image_tensors:
            # Process image tensor before calculating the clip score
            inputs = self.process_img(tensor)
            features = self.clip_model.get_image_features(pixel_values=inputs)

            clip_embeddings.append(features)

        return clip_embeddings

    def calculate_similarity(self, image_tensors):
        processed = [self.process_img(image_tensor)
                     for image_tensor in image_tensors]
        processed = torch.cat(processed)

        # Get normalized image features with gradient tracking
        features = self.clip_model.get_image_features(pixel_values=processed)
        features = F.normalize(features, dim=-1)  # Shape: [B, D]

        # Compute similarity matrix
        sim_matrix = features @ features.T  # Gradient will flow through this

        # Ensure mean() is used instead of .item() to allow gradients
        return sim_matrix.mean(), features

    def calculate_spread(self, image_tensors):
        # Process entire batch at once (maintains gradients)
        processed = [self.process_img(image_tensor)
                     for image_tensor in image_tensors]
        processed = torch.cat(processed, dim=0)

        # Get normalized image features with gradient tracking
        features = self.clip_model.get_image_features(pixel_values=processed)
        features = F.normalize(features, dim=-1)  # Shape: [B, D]

        # Compute pairwise cosine distances using PyTorch operations
        cosine_similarity = torch.mm(features, features.T)  # [B, B]
        pairwise_dists = 1 - cosine_similarity  # Convert to distance

        # Extract upper triangle (excluding diagonal)
        n = features.size(0)
        rows, cols = torch.triu_indices(n, n, offset=1)
        upper_triangle = pairwise_dists[rows, cols]

        # Calculate mean spread while maintaining gradient flow
        mean_spread = upper_triangle.mean()

        return mean_spread
    
    def run(self, prompt):
        # Encode prompt
        prompt_embeds, negative_prompt_embeds = self.pipeline_base.encode_prompt(
            prompt,
            self.device,
            self.num_images_per_prompt,
            self.do_classifier_free_guidance,
            self.negative_prompt
        )
    
        # Retrieve the SOS token, the condition, and uncondition
        SOS_token = prompt_embeds[:, 0, :]
        context_condition = torch.nn.Parameter(prompt_embeds[:, 1:, :])
        context_uncondition = torch.nn.Parameter(negative_prompt_embeds[:, 1:, :])
    
        # We want to optimise the context_condition and context_uncondition
        # (the prompt_embedding and negative_prompt_embedding without the SOS token)
        context_condition.requires_grad_(True)
        context_uncondition.requires_grad_(True)
    
        # Setup optimizer for the embedding
        optimizer = optim.Adam(
            [context_condition, context_uncondition], lr=self.learning_rate)
    
        # Prepare latent variables
        batch_size = 1
        # spatial dimension of 64x64 (for 512x512 output images)
        latent_shape = (1, 4, 64, 64)
    
        # Initialise CLIPModel
        # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        # model.to(self.device)
        self.clip_model.requires_grad_(False)
        # Initialize variables to track best performance
        best_objective_value = float('-inf')
        best_context_condition = None
        best_context_uncondition = None
    
        # Starting iteration
        start_iteration = 0
    
        # Initialise a list to store objective values over iterations
        objective_history = []
    
        init_latents_base = []
        init_latents_lora = []
        for seed in SEEDS:
            generator_base = torch.Generator(self.device).manual_seed(seed)
            generator_lora = torch.Generator(self.device).manual_seed(seed)
            init_latents_base.append(torch.randn(
                latent_shape, generator=generator_base, device=self.device))
            init_latents_lora.append(torch.randn(
                latent_shape, generator=generator_lora, device=self.device))
    
        for iter in range(start_iteration, self.num_optimization_steps):
            optimizer.zero_grad()
    
            # get prompt embeddings using optimised context_condition and context_uncondition
            prompt_embeds = self.get_text_embedding(
                context_condition, context_uncondition, SOS_token)
    
            # Generate num_images image tensor for base SD model
            image_tensors_base = []
            for j in range(self.num_images):
                # The initial latent (which is the initial noise) is copmuted to be used for every image generation
                init_latent_base = init_latents_base[j].clone()
    
                # Prepare timesteps
                self.scheduler_base.set_timesteps(self.num_inference_steps, device=self.device)
                timesteps_base = self.scheduler_base.timesteps
    
                latents_base = self.do_denoise(self.pipeline_base, scheduler=self.scheduler_base, timesteps=timesteps_base,
                                          latents=init_latent_base, prompt_embeds=prompt_embeds, guidance_scale=self.guidance_scale)
                image_tensor_base = self.latent2img_tensor(latents_base, self.pipeline_base)
                image_tensors_base.append(image_tensor_base)
    
            # Generate num_images image tensor for LoRA model
            image_tensors_lora = []
            for j in range(self.num_images):
                # The initial latent (which is the initial noise) is copmuted to be used for every image generation
                init_latent_lora = init_latents_lora[j].clone()
    
                # Prepare timesteps
                self.scheduler_lora.set_timesteps(self.num_inference_steps, device=self.device)
                timesteps_lora = self.scheduler_lora.timesteps
    
                latents_lora = self.do_denoise(self.pipeline_lora, scheduler=self.scheduler_lora, timesteps=timesteps_lora,
                                          latents=init_latent_lora, prompt_embeds=prompt_embeds, guidance_scale=self.guidance_scale)
                image_tensor_lora = self.latent2img_tensor(latents_lora, self.pipeline_lora)
                image_tensors_lora.append(image_tensor_lora)
    
            # Calculating score
            intra_model_consistency_1, base_features = self.calculate_similarity(
                image_tensors_base)
            intra_model_consistency_2, lora_features = self.calculate_similarity(
                image_tensors_lora)
    
            sim_matrix = base_features @ lora_features.T
            inter_model_similarity = sim_matrix.mean()
    
            base_spread = self.calculate_spread(image_tensors_base)
            lora_spread = self.calculate_spread(image_tensors_lora)
    
            objective_value = -ALPHA * lora_spread + BETA * \
                (1 - inter_model_similarity) + GAMMA * base_spread
            objective_value = torch.tanh(objective_value)
    
            objective_history.append(objective_value.item())
    
            # Update best values and save images if current objective is better
            if objective_value.item() > best_objective_value:
                print(
                    f"New best objective in step {iter}, Objective: {objective_value.item():.4f}")
                best_objective_value = objective_value.item()
                best_context_condition = context_condition.detach().clone()
                best_context_uncondition = context_uncondition.detach().clone()
    
            # Back propagation & Gradient descent
            # We want to maximise (S_intra - S_inter). Gradient descent minimises loss function
            (-objective_value).backward()
            optimizer.step()
    
        # Generate results
        print(f"Best objective value: {best_objective_value}")
    
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(range(self.num_optimization_steps), objective_history, marker='o')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective Value")
        ax.set_title("Objective Value over Time")
        ax.grid(True)
        
        history_path = os.path.join("results", "step2_objective_history.png")
        fig.savefig(history_path, bbox_inches="tight")
        plt.close(fig)
    
        prompt_embeds = self.get_text_embedding(best_context_condition, best_context_uncondition, SOS_token)
    
        image_tensors_base = []
        for j in range(self.num_images):
            # The initial latent (which is the initial noise) is copmuted to be used for every image generation
            init_latent_base = init_latents_base[j].clone()
    
            # Prepare timesteps
            self.scheduler_base.set_timesteps(self.num_inference_steps, device=self.device)
            timesteps_base = self.scheduler_base.timesteps
    
            latents_base = self.do_denoise(self.pipeline_base, scheduler=self.scheduler_base, timesteps=timesteps_base,
                                      latents=init_latent_base, prompt_embeds=prompt_embeds, guidance_scale=self.guidance_scale)
            image_tensor_base = self.latent2img_tensor(latents_base, self.pipeline_base)
            image_tensors_base.append(image_tensor_base)
    
        image_tensors_lora = []
        for j in range(self.num_images):
            # The initial latent (which is the initial noise) is copmuted to be used for every image generation
            init_latent_lora = init_latents_lora[j].clone()
    
            # Prepare timesteps
            self.scheduler_lora.set_timesteps(self.num_inference_steps, device=self.device)
            timesteps_lora = self.scheduler_lora.timesteps
    
            latents_lora = self.do_denoise(self.pipeline_lora, scheduler=self.scheduler_lora, timesteps=timesteps_lora,
                                      latents=init_latent_lora, prompt_embeds=prompt_embeds, guidance_scale=self.guidance_scale)
            image_tensor_lora = self.latent2img_tensor(latents_lora, self.pipeline_lora)
            image_tensors_lora.append(image_tensor_lora)
    
        pils_base = [transforms.ToPILImage()(image_tensor.squeeze(0))
                     for image_tensor in image_tensors_base]
        pils_lora = [transforms.ToPILImage()(image_tensor.squeeze(0))
                     for image_tensor in image_tensors_lora]
            
        fig, axes = plt.subplots(2, len(pils_base), figsize=(15, 6))
        fig.suptitle(f"Step 2 Output -> Best Objective: {best_objective_value:.4f}", fontsize=16)
        
        axes[0, 0].text(-0.1, 0.5, "Base SD", transform=axes[0,0].transAxes,
                        fontsize=14, fontweight="bold", va="center")
        axes[1, 0].text(-0.1, 0.5, "LoRA",    transform=axes[1,0].transAxes,
                        fontsize=14, fontweight="bold", va="center")
        
        for i, img in enumerate(pils_base):
            axes[0, i].imshow(img)
            axes[0, i].axis("off")
        
        for i, img in enumerate(pils_lora):
            axes[1, i].imshow(img)
            axes[1, i].axis("off")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, left=0.15)
        
        out_path = os.path.join("results", "step_2_out.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        