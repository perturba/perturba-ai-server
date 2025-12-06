import torch
import torch.nn.functional as F
from typing import Callable, Optional, List, Dict

import numpy as np
import cv2
import mediapipe as mp

from diffusers import StableDiffusionInpaintPipeline, AutoencoderKL
from facenet_pytorch import InceptionResnetV1

from torchvision.transforms import ToPILImage
import torchvision.transforms as T
from PIL import Image
from pytorch_msssim import ssim

import math

class PGDAttackVAEInpaint:
    """
    - VAE encoder 기반 PGD 공격
    - MediaPipe 기반 얼굴 binary mask 생성
    - Stable Diffusion v2 Inpainting
    - Mask 기반 FaceNet identity similarity 계산
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        device: Optional[torch.device] = None,
        iters: int = 20,
        dtype=torch.float32,
        hf_auth_token: Optional[str] = None,
        disable_safety: bool = True,
        verbose: bool = True,
    ):
        self.device = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype
        # self.eps = eps
        # self.alpha = alpha
        self.iters = iters
        self.verbose = verbose

        if verbose:
            print(f"[Init] Loading SD Inpainting Pipeline: {model_id}")

        pipe_kwargs = {"torch_dtype": self.dtype}
        if hf_auth_token is not None:
            pipe_kwargs["use_auth_token"] = hf_auth_token

        self.pipe: StableDiffusionInpaintPipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, **pipe_kwargs
        ).to(self.device)

        if disable_safety and hasattr(self.pipe, "safety_checker"):
            self.pipe.safety_checker = None

        # VAE encoder (surrogate)
        self.vae: AutoencoderKL = self.pipe.vae.to(self.device)
        for p in self.vae.parameters():
            p.requires_grad = False

        # FaceNet for identity
        if verbose:
            print("[Init] Loading FaceNet (VGGFace2)...")
        self.facenet = InceptionResnetV1(pretrained="vggface2").to(self.device).eval()

        # helpers
        self.to_pil = ToPILImage()

        if verbose:
            print("[Init] Done.\n")

        # loss_fn

        self.loss_fn = l2_loss
        self.ssim = ssim_loss

    # -------------------------------------------------------------------------
    # VAE encoder
    # -------------------------------------------------------------------------
    def _image_to_latent(self, images: torch.Tensor) -> torch.Tensor:

        images = images.to(self.device).to(self.dtype)
        imgs_in = 2.0 * images - 1.0  # [-1,1]

        enc = self.vae.encode(imgs_in)
        latent_dist = enc.latent_dist

        if hasattr(latent_dist, "rsample"):
            latents = latent_dist.rsample()
        else:
            latents = latent_dist.mean

        latents = latents * self.vae.config.scaling_factor
        return latents

    def vae_encoding(self, images: torch.Tensor) -> torch.Tensor:

        images = images.to(self.device).to(self.dtype)
        imgs_in = 2.0 * images - 1.0  # [-1,1]

        enc = self.vae.encode(imgs_in)
        latent_dist = enc.latent_dist

        if hasattr(latent_dist, "rsample"):
            latents = latent_dist.rsample()
        else:
            latents = latent_dist.mean

        latents = latents * self.vae.config.scaling_factor
        return latents

    # -------------------------------------------------------------------------
    # MediaPipe Face Binary Mask
    # -------------------------------------------------------------------------
    def _get_face_binary_mask(self, pil_img: Image.Image) -> Image.Image:

        img_rgb = np.array(pil_img)
        h, w, _ = img_rgb.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        with mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ) as face_detection:
            results = face_detection.process(img_rgb)

            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box

                    x1 = int(max(0, bbox.xmin * w))
                    y1 = int(max(0, bbox.ymin * h))
                    x2 = int(min(w, (bbox.xmin + bbox.width) * w))
                    y2 = int(min(h, (bbox.ymin + bbox.height) * h))

                    mask[y1:y2, x1:x2] = 255

                kernel = np.ones((15, 15), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)

            else:
                mask[:, :] = 255

        return Image.fromarray(mask)

    # -------------------------------------------------------------------------
    # Mask-based face embedding
    # -------------------------------------------------------------------------
    def _masked_face_embedding(
        self,
        pil_img: Image.Image,
        mask_img: Image.Image
    ) -> Optional[torch.Tensor]:

        img = np.array(pil_img)
        mask = np.array(mask_img)
        
        if mask.ndim == 3:
            mask = mask[..., 0]
            
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        face = img.copy()
        face[mask == 0] = 0

        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None

        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()

        face_crop = face[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        face_tensor = transform(face_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.facenet(face_tensor)

        return emb

    # -------------------------------------------------------------------------
    # Mask-based identity similarity
    # -------------------------------------------------------------------------
    def _compute_identity_similarity_masked(
        self,
        img1: Image.Image,
        img2: Image.Image,
        mask1: Image.Image,
        mask2: Image.Image
    ) -> Optional[float]:

        emb1 = self._masked_face_embedding(img1, mask1)
        emb2 = self._masked_face_embedding(img2, mask2)

        if emb1 is None or emb2 is None:
            return None

        cos = F.cosine_similarity(emb1, emb2).item()
        return cos

    # -------------------------------------------------------------------------
    # Main Attack
    # -------------------------------------------------------------------------
    def attack(
        self,
        original_images: torch.Tensor,
        eps = 8.0,
        prompt: str = "",
        gen_strength: float = 0.7,
        gen_guidance_scale: float = 7,
        gen_steps: int = 30,
    ) -> Dict[str, object]:

        eps = eps / 255
        alpha = eps / 4
        assert original_images.ndim == 4, "Input must be (B,3,H,W)"
        B = original_images.shape[0]

        original_images = original_images.to(self.device).to(self.dtype)

        with torch.no_grad():
            orig_latents = self._image_to_latent(original_images).detach()

        adv_images = original_images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, 0.0, 1.0)

        masks = []
        
        for b in range(B):
            orig = original_images[b].detach().cpu().clamp(0, 1)
            pil_orig = self.to_pil(orig)
            mask = self._get_face_binary_mask(pil_orig)
            masks.append(mask)

        if self.verbose:
            print(f"[PGD] Start Attack | iters={self.iters}, eps={eps}")

        for i in range(self.iters):
            
            adv_images.requires_grad_(True)
            pert_latents = self._image_to_latent(adv_images)
            
            # l2_l = self.loss_fn(orig_latents, pert_latents).mean()
            # ssim_l = self.ssim(orig_latents, pert_latents)

            mask = masks[0]
            mask = np.array(mask) / 255.0
            
            if mask.ndim == 3:
                mask = mask[..., 0]
            if mask.shape[:2] != orig_latents.shape[:2]:
                mask = cv2.resize(mask, (orig_latents.shape[2], orig_latents.shape[3]), interpolation=cv2.INTER_NEAREST)

            mask_tensor = torch.from_numpy(mask).to(self.device).to(self.dtype)  # (H, W)
            mask_tensor = mask_tensor.unsqueeze(0).repeat(4,1,1)  # (C, H, W)

            masked_orig = orig_latents * mask_tensor
            masked_pert = pert_latents * mask_tensor

            l2_ml = self.loss_fn(masked_orig, masked_pert).mean()
            ssim_ml = self.ssim(masked_orig, masked_pert)
            
            # total_loss = (-l2_l) + (-l2_ml) + ssim_l + ssim_ml
            total_loss = (-l2_ml) + ssim_ml
            total_loss.backward()
            
            alpha_t = alpha * (1 + math.cos(math.pi * i / self.iters)) / 2
            alpha_t = (alpha / 1000) + 0.5 * (alpha - (alpha / 1000)) * (1 + math.cos(math.pi * i / self.iters))
            
            with torch.no_grad():
                adv_images = adv_images + alpha_t * adv_images.grad.sign()
                delta = torch.clamp(adv_images - original_images, -eps, eps)
                adv_images = torch.clamp(original_images + delta, 0.0, 1.0).detach()
            
            if self.verbose and (i % 10 == 0 or i == self.iters - 1):
                print(f"[PGD] iter {i+1}/{self.iters} | loss={total_loss.item():.5f}")
        
        perturbed_images = adv_images.detach().cpu()
        
        gen_adv_list = []
        gen_ori_list = []
        similarities = []
        
        for b in range(B):
        
            orig = original_images[b].detach().cpu().clamp(0, 1)
            adv = perturbed_images[b].clamp(0, 1)

            pil_orig = self.to_pil(orig)
            pil_adv = self.to_pil(adv)
            
            # with torch.no_grad():
            #     out = self.pipe(
            #         prompt=prompt,
            #         image=pil_orig,
            #         mask_image=mask,
            #         strength=gen_strength,
            #         guidance_scale=gen_guidance_scale,
            #         num_inference_steps=gen_steps,
            #     )

            #     gen_ori = out.images[0]
            #     gen_ori_list.append(gen_ori)

            with torch.no_grad():
                out = self.pipe(
                    prompt=prompt,
                    image=pil_adv,
                    mask_image=mask,
                    strength=gen_strength,
                    guidance_scale=gen_guidance_scale,
                    num_inference_steps=gen_steps,
                )

                gen_adv = out.images[0]
                gen_adv_list.append(gen_adv)

            sim = self._compute_identity_similarity_masked(
                pil_orig, gen_adv, mask, mask
            )
            similarities.append(sim)
            
            # sim_ori = self._compute_identity_similarity_masked(
            #     pil_orig, gen_ori, mask, mask
            # )
            # similarities.append(sim_ori)

        result = {
            "original_images": original_images.detach().cpu(),
            "perturbed_images": perturbed_images,
            "masks": masks,
            #"gen_orig" : gen_ori_list,
            "gen_adv": gen_adv_list,
            "identity_similarity": similarities
        }

        return result

def l2_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 (mean squared error) loss between x and y.
    Args:
        x, y: [B, C, H, W] tensors
    Returns:
        scalar tensor (mean squared error)
    """
    return F.mse_loss(x, y)

def ssim_loss(x,y):
    
    return 1.0 - ssim(x, y, data_range=1.0, size_average=True)
