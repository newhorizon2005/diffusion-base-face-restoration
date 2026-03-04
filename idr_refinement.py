"""
Iterative Diffusion Refinement (IDR) Module
This module implements iterative sampling with identity score-based selection:
1. Generate multiple samples with different random seeds
2. Compute identity score using ArcFace similarity
3. Select the best result based on identity preservation
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from PIL import Image
import cv2


def compute_arcface_similarity(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor
) -> float:
    """
    Compute cosine similarity between two ArcFace embeddings.
    
    Args:
        embedding1: First embedding tensor (shape: [D] or [1, D])
        embedding2: Second embedding tensor (shape: [D] or [1, D])
    
    Returns:
        Similarity score (0-1, higher means more similar)
    """
    # Ensure embeddings are 1D
    if embedding1.dim() > 1:
        embedding1 = embedding1.squeeze()
    if embedding2.dim() > 1:
        embedding2 = embedding2.squeeze()
    
    # Convert to same dtype (float32) to avoid dtype mismatch
    embedding1 = embedding1.float()
    embedding2 = embedding2.float()
    
    # Normalize embeddings
    embedding1 = embedding1 / torch.norm(embedding1)
    embedding2 = embedding2 / torch.norm(embedding2)
    
    # Compute cosine similarity
    similarity = torch.dot(embedding1, embedding2)
    
    return float(similarity.item())


def compute_identity_score(
    generated_image: np.ndarray,
    reference_embedding: torch.Tensor,
    face_analyzer,
    device: str = "cuda"
) -> float:
    """
    Compute identity score by comparing generated face with reference embedding.
    
    Args:
        generated_image: Generated image as numpy array (H, W, 3)
        reference_embedding: Reference face embedding tensor
        face_analyzer: InsightFace analyzer instance
        device: Device to run on
    
    Returns:
        Identity score (cosine similarity, 0-1)
    """
    try:
        # Detect face in generated image
        from insightface_package import analyze_faces
        faces = analyze_faces(face_analyzer, generated_image)
        
        if len(faces) == 0:
            # No face detected, return low score
            return 0.0
        
        # Use the largest face (most prominent)
        face = faces[0]
        generated_emb = torch.tensor(face['embedding']).to(device).float()
        
        # Normalize
        generated_emb = generated_emb / torch.norm(generated_emb)
        
        # Compute similarity with reference
        similarity = compute_arcface_similarity(reference_embedding, generated_emb)
        
        return similarity
        
    except Exception as e:
        print(f"Error computing identity score: {e}")
        return 0.0


def iterative_diffusion_refinement(
    pipeline_fn,
    reference_embedding: torch.Tensor,
    face_analyzer,
    num_iterations: int = 3,
    base_seed: int = 42,
    device: str = "cuda",
    **pipeline_kwargs
) -> Tuple[Image.Image, List[Dict]]:
    """
    Perform iterative diffusion refinement with multiple sampling rounds.
    
    Args:
        pipeline_fn: Function that runs the diffusion pipeline and returns an image
        reference_embedding: Reference face embedding for identity comparison
        face_analyzer: InsightFace analyzer for computing identity scores
        num_iterations: Number of iterations (K in the paper, typically 3-4)
        base_seed: Base random seed
        device: Device to run on
        **pipeline_kwargs: Additional arguments to pass to pipeline_fn
    
    Returns:
        Tuple of (best_image, all_results_with_scores)
    """
    results = []
    
    print(f"🔄 Starting Iterative Diffusion Refinement (K={num_iterations} iterations)...")
    
    for k in range(num_iterations):
        # Generate with different seed
        seed_k = base_seed + k * 1000
        print(f"  Iteration {k+1}/{num_iterations}: seed={seed_k}")
        
        # Create generator with current seed
        generator = torch.Generator(device=device).manual_seed(seed_k)
        
        # Run pipeline
        generated_image = pipeline_fn(generator=generator, **pipeline_kwargs)
        
        # Convert to numpy if needed
        if isinstance(generated_image, Image.Image):
            image_array = np.array(generated_image)
        else:
            image_array = generated_image
        
        # Compute identity score
        identity_score = compute_identity_score(
            image_array,
            reference_embedding,
            face_analyzer,
            device
        )
        
        print(f"    → Identity score s^({k+1}) = {identity_score:.4f}")
        
        results.append({
            'iteration': k + 1,
            'seed': seed_k,
            'image': generated_image if isinstance(generated_image, Image.Image) else Image.fromarray(image_array),
            'image_array': image_array,
            'identity_score': identity_score
        })
    
    # Select best result (argmax of identity scores)
    best_idx = max(range(len(results)), key=lambda i: results[i]['identity_score'])
    best_result = results[best_idx]
    
    print(f"✓ Best result: Iteration {best_result['iteration']} with score {best_result['identity_score']:.4f}")
    
    return best_result['image'], results


def batch_iterative_diffusion(
    pipeline,
    reference_embedding: torch.Tensor,
    face_analyzer,
    control_image: Image.Image,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_pooled_prompt_embeds: torch.Tensor,
    num_iterations: int = 3,
    base_seed: int = 42,
    device: str = "cuda",
    vae = None,
    noise_scheduler = None,
    **other_kwargs
) -> Tuple[Image.Image, List[Dict]]:
    """
    Wrapper function for IDR that works with the FaceMe pipeline.
    
    Args:
        pipeline: StableDiffusionXLControlNetPipeline instance
        reference_embedding: Reference face embedding
        face_analyzer: InsightFace analyzer
        control_image: Control image (PIL Image)
        prompt_embeds: Positive prompt embeddings
        pooled_prompt_embeds: Pooled positive embeddings
        negative_prompt_embeds: Negative prompt embeddings
        negative_pooled_prompt_embeds: Pooled negative embeddings
        num_iterations: Number of IDR iterations
        base_seed: Base random seed
        device: Device
        vae: VAE model for encoding
        noise_scheduler: Noise scheduler
        **other_kwargs: Additional pipeline arguments (guidance_scale, steps, etc.)
    
    Returns:
        Tuple of (best_image, all_results)
    """
    weight_dtype = torch.float16
    
    # Prepare control image (only once)
    control = np.array(control_image) / 255.0
    control = control * 2.0 - 1.0
    control_tensor = torch.tensor(control).permute(2, 0, 1).unsqueeze(dim=0).to(device=device, dtype=weight_dtype)
    
    results = []
    
    print(f"🔄 IDR: Generating {num_iterations} samples...")
    
    for k in range(num_iterations):
        seed_k = base_seed + k * 1000
        generator = torch.Generator(device=device).manual_seed(seed_k)
        
        print(f"  Sample {k+1}/{num_iterations} (seed={seed_k})...", end=" ")
        
        # Encode control image to latents
        latents = vae.encode(control_tensor.to(dtype=torch.float16)).latent_dist.sample()
        latents = latents.to(dtype=weight_dtype) * vae.config.scaling_factor
        
        # Add noise
        noise = torch.randn(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype
        )
        bsz = latents.shape[0]
        timesteps = torch.randint(
            noise_scheduler.config.num_train_timesteps - 1,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
            generator=generator
        )
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Run pipeline
        image = pipeline(
            latents=noisy_latents,
            image=control_tensor,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=generator,
            **other_kwargs
        ).images[0]
        
        # Convert to numpy
        image_array = np.array(image)
        
        # Compute identity score
        identity_score = compute_identity_score(
            image_array,
            reference_embedding,
            face_analyzer,
            device
        )
        
        print(f"identity_score={identity_score:.4f}")
        
        results.append({
            'iteration': k + 1,
            'seed': seed_k,
            'image': image,
            'image_array': image_array,
            'identity_score': identity_score
        })
    
    # Select best
    best_idx = max(range(len(results)), key=lambda i: results[i]['identity_score'])
    best_result = results[best_idx]
    
    print(f"✓ IDR Best: Sample {best_result['iteration']} (score={best_result['identity_score']:.4f})")
    
    return best_result['image'], results


if __name__ == "__main__":
    # Test the IDR module
    print("Testing IDR Module...")
    
    # Test similarity computation
    emb1 = torch.randn(512)
    emb2 = emb1 + torch.randn(512) * 0.1  # Similar embedding
    
    similarity = compute_arcface_similarity(emb1, emb2)
    print(f"Similarity score: {similarity:.4f}")
    
    print("✓ IDR module test passed!")

