import os
import torch
import argparse
import numpy as np
import gradio as gr
from tqdm import tqdm
from PIL import Image
from typing import List
from huggingface_hub import hf_hub_download

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
)
from load_photomaker import load_photomaker
from idencoder import PhotoMakerIDEncoder, Mix
from wavelet_color_fix import wavelet_reconstruction
from insightface_package import FaceAnalysis2, analyze_faces
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel, StableDiffusionXLControlNetPipeline
from transformers import AutoTokenizer, PretrainedConfig, CLIPImageProcessor

# Import MRIA and IDR modules (innovation points)
from mria_quality_score import compute_quality_scores, aggregate_embeddings_with_quality
from idr_refinement import batch_iterative_diffusion, compute_identity_score


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str = None,
                                               subfolder: str = "text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def encode_prompt(text_encoders, text_input_ids_list=None):
    prompt_embeds_list = []
    for i, text_encoder in enumerate(text_encoders):
        prompt_embeds = text_encoder(
            text_input_ids_list[i].to(text_encoder.device), output_hidden_states=True, return_dict=False
        )
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def token_prompt(text_tokenizer, prompt):
    text_tokens = []
    for i, tokenizer in enumerate(text_tokenizer):
        tokens = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt").input_ids
        text_tokens.append(tokens)

    return text_tokens


def prepare_text_encoder(args):
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        use_fast=False,
    )
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path,
                                                                      subfolder="text_encoder_2")

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
    )
    text_tokenizer = [tokenizer_one, tokenizer_two]
    text_encoder = [text_encoder_one, text_encoder_two]

    tag_token = tokenizer_one.encode(args.key_word)[1]

    return text_tokenizer, text_encoder, tag_token


def prepare_text_emb(text_tokenizer, text_encoders, prompt, tag_token, return_index=False):
    tokens = token_prompt(text_tokenizer, prompt)
    if return_index:
        index = torch.where(tokens[0] == tag_token)[1]
    else:
        index = None

    embeds, pooled_embeds = encode_prompt(text_encoders, tokens)

    return index, embeds, pooled_embeds


def prepare(args):
    text_tokenizer, text_encoder, tag_token = prepare_text_encoder(args)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None)
    unet = OriginalUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
    )
    id_encoder_clip = PhotoMakerIDEncoder()
    # photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
    # 在这里使用本地直接加载
    id_encoder_clip, unet = load_photomaker("models/TencentARC--PhotoMaker/photomaker-v1.bin", clip_id_encoder=id_encoder_clip, unet=unet)
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    mix = Mix()
    mix.from_pretrained(args.mix_path)

    app = FaceAnalysis2(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(512, 512))

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    mix.requires_grad_(False)
    id_encoder_clip.requires_grad_(False)
    controlnet.requires_grad_(False)

    weight_dtype = torch.float16
    device = args.device
    vae.to(device, dtype=torch.float16)
    unet.to(device, dtype=weight_dtype)
    mix.to(device, dtype=weight_dtype)
    id_encoder_clip.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)

    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        controlnet=controlnet,
        unet=unet,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = noise_scheduler
    pipeline = pipeline.to(device=device)
    return pipeline, id_encoder_clip, app, vae, noise_scheduler, mix, text_tokenizer, text_encoder, tag_token


def main(args):
    pipeline, id_encoder_clip, app, vae, noise_scheduler, mix, text_tokenizer, text_encoder, tag_token = prepare(args)

    ref_folder = "data/ref"
    img_ext = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

    image_files = [
        os.path.join(ref_folder, f)
        for f in os.listdir(ref_folder)
        if os.path.splitext(f)[-1].lower() in img_ext
    ]

    ref_image = []
    for path in image_files:
        try:
            img = Image.open(path).convert("RGB")
            ref_image.append(img)
        except Exception as e:
            print(f"error pic:{path},because:{e}")

    print(f"loaded {len(ref_image)} pics as reference")

    @torch.no_grad()
    def process(
            control_img: Image.Image,
            pos_prompt: str,
            neg_prompt: str,
            num_samples: int,
            strength: float,
            cfg_scale: float,
            steps: int,
            use_color_fix: bool,
            seed: int,
            use_mria: bool = True,
            use_idr: bool = True,
            idr_iterations: int = 3,
    ) -> List[np.ndarray]:

        nonlocal  ref_image

        device = args.device

        mask_index, pos_prompt_embeds, pos_pooled_prompt_embeds = prepare_text_emb(text_tokenizer, text_encoder,
                                                                                   pos_prompt, tag_token,
                                                                                   return_index=True)
        _, neg_prompt_embeds, neg_pooled_prompt_embeds = prepare_text_emb(text_tokenizer, text_encoder, neg_prompt,
                                                                          tag_token, return_index=False)

        generator = torch.Generator(device=device).manual_seed(seed)

        print(
            f"control image shape={control_img.size}\n"
            f"num_samples={num_samples}\n"
            f"strength={strength}\n"
            f"cdf scale={cfg_scale}, steps={steps}, use_color_fix={use_color_fix}\n"
            f"seed={seed}\n"
        )
        clip_processor = CLIPImageProcessor()

        control_img = control_img.resize((512, 512))
        h, w = control_img.height, control_img.width

        weight_dtype = torch.float16

        ref_id_embs = []
        ref_clip_embs = []
        quality_scores = []  # For MRIA
        face_infos = []  # Store face info for later use

        # 强制类型转换list
        if isinstance(ref_image, Image.Image):
            ref_image = [ref_image]

        print(f"📊 Processing {len(ref_image)} reference faces with MRIA quality scoring...")
        
        for i, ref in enumerate(ref_image):
            ref_array = np.array(ref)
            try:
                detect_face = analyze_faces(app, ref_array)[0]
            except:
                raise ValueError(f"Can't detect face: {i}")

            # ===== MRIA Innovation Point 1: Quality Score Computation =====
            if use_mria:
                quality_score_dict = compute_quality_scores(
                    ref_array, 
                    detect_face,
                    weights={'sharpness': 0.5, 'pose': 0.3, 'illumination': 0.2}
                )
                overall_quality = quality_score_dict['overall']
                quality_scores.append(overall_quality)
                print(f"  Reference {i+1}: Quality={overall_quality:.3f} "
                      f"(sharp={quality_score_dict['sharpness']:.3f}, "
                      f"pose={quality_score_dict['pose']:.3f}, "
                      f"illum={quality_score_dict['illumination']:.3f})")
            else:
                quality_scores.append(1.0)  # Equal weights if MRIA disabled

            ref_emb = detect_face['embedding']
            ref_emb = torch.tensor(ref_emb).to(device)
            ref_emb = ref_emb / torch.norm(ref_emb, dim=0, keepdim=True)  # normalize embedding
            ref_id_embs.append(ref_emb.unsqueeze(dim=0).to(dtype=weight_dtype))

            crop_face = Image.fromarray(ref_array).crop(detect_face["bbox"])
            crop_face = clip_processor(crop_face)["pixel_values"][0]
            crop_face = torch.tensor(crop_face).to(device).unsqueeze(dim=0)

            ref_clip_emb = id_encoder_clip(crop_face)
            ref_clip_embs.append(ref_clip_emb.to(dtype=weight_dtype))
            
            face_infos.append(detect_face)

        # ===== MRIA Innovation Point 2: Quality-Weighted Aggregation =====
        if use_mria and len(ref_id_embs) > 1:
            print(f"🔬 MRIA: Aggregating {len(ref_id_embs)} embeddings with quality weighting...")
            
            # Aggregate ID embeddings with quality weighting
            ref_id_embs_aggregated = aggregate_embeddings_with_quality(
                ref_id_embs, 
                quality_scores, 
                method='weighted_average'
            )
            
            # Aggregate CLIP embeddings with quality weighting
            ref_clip_embs_aggregated = aggregate_embeddings_with_quality(
                ref_clip_embs,
                quality_scores,
                method='weighted_average'
            )
            
            embs = mix(clip_emb=ref_clip_embs_aggregated, id_emb=ref_id_embs_aggregated)
            
            # Store best reference embedding for IDR
            best_ref_idx = quality_scores.index(max(quality_scores))
            best_ref_embedding = ref_id_embs[best_ref_idx].squeeze()
        else:
            # Fallback to original concatenation if MRIA disabled
            ref_id_embs = torch.cat(ref_id_embs, dim=0)
            ref_clip_embs = torch.cat(ref_clip_embs, dim=0)
            embs = mix(clip_emb=ref_clip_embs, id_emb=ref_id_embs)
            best_ref_embedding = ref_id_embs[0].squeeze() if len(ref_id_embs) > 0 else ref_id_embs.squeeze()
        pref = pos_prompt_embeds[:, :mask_index, :].clone().to(device)
        sufx = pos_prompt_embeds[:, mask_index + 1:, :].clone().to(device)
        _pos_prompt_embeds = torch.cat([pref, embs.unsqueeze(dim=0), sufx], dim=1)[:, :77, :]

        preds = []
        
        # ===== IDR Innovation Point: Iterative Diffusion Refinement =====
        if use_idr:
            print(f"🔄 Using Iterative Diffusion Refinement (IDR) with K={idr_iterations} iterations...")
            
            for sample_idx in range(num_samples):
                current_seed = seed + sample_idx * 10000
                
                # Run IDR: generate multiple samples and select best based on identity score
                best_image, all_results = batch_iterative_diffusion(
                    pipeline=pipeline,
                    reference_embedding=best_ref_embedding,
                    face_analyzer=app,
                    control_image=control_img,
                    prompt_embeds=_pos_prompt_embeds,
                    pooled_prompt_embeds=pos_pooled_prompt_embeds,
                    negative_prompt_embeds=neg_prompt_embeds,
                    negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                    num_iterations=idr_iterations,
                    base_seed=current_seed,
                    device=device,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    guidance_scale=float(cfg_scale),
                    controlnet_conditioning_scale=float(strength),
                    num_inference_steps=int(steps),
                )
                
                # Apply color fix if enabled
                image_array = np.array(best_image)
                control = np.array(control_img) / 255.0
                control = control * 2.0 - 1.0
                control_tensor = torch.tensor(control).permute(2, 0, 1).unsqueeze(dim=0)
                
                sr = torch.tensor((image_array / 255.) * 2. - 1.).permute(2, 0, 1).unsqueeze(dim=0)
                if use_color_fix:
                    sr = wavelet_reconstruction(sr.to(torch.float32).cpu(), control_tensor.to(torch.float32).cpu())
                
                sr = sr.squeeze().permute(1, 2, 0) * 127.5 + 127.5
                sr = sr.cpu().numpy().clip(0, 255).astype(np.uint8)
                
                # Save result
                from datetime import datetime
                current_time = datetime.now()
                time_str = current_time.strftime("%Y%m%d%H%M%S")
                Image.fromarray(sr).save(os.path.join("./app_result", time_str + "_idr_best.png"))
                
                # Also save all IDR iterations for comparison (optional)
                for result in all_results:
                    iter_img = np.array(result['image'])
                    save_name = f"{time_str}_idr_iter{result['iteration']}_score{result['identity_score']:.3f}.png"
                    Image.fromarray(iter_img).save(os.path.join("./app_result", save_name))
                
                preds.append((np.array(sr), f"result_{sample_idx}_idr_best.png"))
        
        else:
            # Original single-pass generation (no IDR)
            print("⚡ Using standard single-pass generation...")
            
            for i in tqdm(range(num_samples)):
                control = np.array(control_img) / 255.0
                control = control * 2.0 - 1.0
                control = torch.tensor(control).permute(2, 0, 1).unsqueeze(dim=0).to(device=device, dtype=weight_dtype)
                latents = vae.encode(control.to(dtype=torch.float16)).latent_dist.sample()
                latents = latents.to(dtype=weight_dtype) * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(noise_scheduler.config.num_train_timesteps - 1,
                                          noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                image = pipeline(
                    latents=noisy_latents,
                    image=control,
                    guidance_scale=float(cfg_scale),
                    controlnet_conditioning_scale=float(strength),
                    prompt_embeds=_pos_prompt_embeds,
                    pooled_prompt_embeds=pos_pooled_prompt_embeds,
                    negative_prompt_embeds=neg_prompt_embeds,
                    negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                    num_inference_steps=int(steps),
                    generator=generator,
                ).images[0]

                sr = torch.tensor((np.array(image) / 255.) * 2. - 1.).permute(2, 0, 1).unsqueeze(dim=0)
                if use_color_fix:
                    sr = wavelet_reconstruction(sr.to(torch.float32).cpu(), control.to(torch.float32).cpu())

                sr = sr.squeeze().permute(1, 2, 0) * 127.5 + 127.5
                sr = sr.cpu().numpy().clip(0, 255).astype(np.uint8)

                from datetime import datetime
                current_time = datetime.now()
                time_str = current_time.strftime("%Y%m%d%H%M%S")

                Image.fromarray(sr).save(os.path.join("./app_result", time_str + ".png"))
                preds.append((np.array(sr), f"result_{i}.png"))
        
        return preds

    MARKDOWN = \
        """
        ## Diffusion-Based Face Restoration with Multi-Reference Identity
        ## 基于扩散模型的定制化人像恢复方法研究与应用
        """

    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            gr.Markdown(MARKDOWN)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources="upload", type="pil")
                run_button = gr.Button(value="Run")
                with gr.Accordion("Options", open=True):
                    pos_prompt = gr.Textbox(label="Positive Prompt",
                                            placeholder="Please enter your prompt (must contain only the keyword: 'face')",
                                            value="a photo of face. High quality facial image, realistic eyes with detailed texture, normal nose, soft expression, smooth skin, skin texture, soft lighting.",
                                            interactive=True)
                    neg_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter your negative prompt here...",
                                            value="painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark,  signature, jpeg artifacts, deformed, lowres, over-smooth",
                                            interactive=True)
                    num_samples = gr.Slider(label="Number Of Samples", minimum=1, maximum=4, value=1, step=1)
                    cfg_scale = gr.Slider(
                        label="Classifier Free Guidance Scale (Set a value larger than 1 to enable it!)", minimum=0.1,
                        maximum=30.0, value=5.0, step=0.1)
                    strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                    use_color_fix = gr.Checkbox(label="Use Color Correction", value=True)
                    seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, value=233)
                    use_mria = gr.Checkbox(
                        label="Enable MRIA (Multi-Reference Identity Aggregation)",
                        value=True,
                        info="Quality-based weighting for multiple reference faces"
                    )
                    use_idr = gr.Checkbox(
                        label="Enable IDR (Iterative Diffusion Refinement)",
                        value=True,
                        info="Generate multiple samples and select best based on identity score"
                    )
                    idr_iterations = gr.Slider(
                        label="IDR Iterations (K)",
                        minimum=2,
                        maximum=5,
                        value=3,
                        step=1,
                        info="Number of sampling iterations (K=3 or 4 recommended)"
                    )
            with gr.Column():
                result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery", scale=2, height="auto")

        inputs = [
            input_image,
            pos_prompt,
            neg_prompt,
            num_samples,
            strength,
            cfg_scale,
            steps,
            use_color_fix,
            seed,
            use_mria,
            use_idr,
            idr_iterations,
        ]

        run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])

    block.launch(share=True, debug=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FaceMe simple example.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="SG161222/RealVisXL_V3.0")
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--mix_path", type=str, default=None)
    parser.add_argument("--pos_prompt", type=str, default='A photo of face.')
    parser.add_argument("--neg_prompt", type=str, default='A photo of face.')
    parser.add_argument("--key_word", type=str, default='face')
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    assert args.key_word in args.pos_prompt

    main(args)
