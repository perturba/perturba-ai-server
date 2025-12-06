# app/services/job_runner.py

import logging
import threading
from io import BytesIO

import requests
from PIL import Image
import torch

from app.config import settings
from app.models.dto import FacePerturbRequest
from app.services.pgd_service import get_pgd_service
from app.clients import spring_client

logger = logging.getLogger(__name__)

# 동시에 돌아가는 Job 개수 제한
concurrency_semaphore = threading.Semaphore(settings.MAX_CONCURRENCY)


def _download_image(url: str) -> Image.Image:
    logger.info(f"Downloading input image from {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def _pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    # 알파 채널 제거 (RGBA, LA 등)
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95, optimize=True)
    buf.seek(0)
    return buf.getvalue()


def _flatten_headers(headers: dict) -> dict:
    flattened = {}
    for k, v in headers.items():
        if isinstance(v, list):
            flattened[k] = ",".join(v)
        else:
            flattened[k] = str(v)
    return flattened


def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    img_tensor = img_tensor.clamp(0, 1)
    np_img = (img_tensor * 255).byte().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(np_img)


def _make_perturbation_vis(orig: torch.Tensor, pert: torch.Tensor) -> Image.Image:
    diff = (pert - orig).abs()
    diff_mean = diff.mean(dim=0)
    max_val = diff_mean.max()
    if max_val > 0:
        diff_mean = diff_mean / max_val
    diff_img = (diff_mean * 255).byte().cpu().numpy()
    pil = Image.fromarray(diff_img, mode="L").convert("RGB")
    return pil


def process_face_perturb_job(req: FacePerturbRequest) -> None:
    service = get_pgd_service()

    with concurrency_semaphore:
        try:
            logger.info(f"[Job {req.jobId}] Start processing.")

            # 1) 입력 이미지 다운로드
            input_img = _download_image(str(req.inputImageUrl))

            # 2) PGD + Inpainting 실행
            result = service.run_attack_raw(
                input_img,
                prompt=req.prompt or "",
                # intensity → eps 매핑을 service 안에서 하든 여기서 하든, 현재 attack 시그니처에 맞춰 조정
            )

            original_images = result["original_images"]
            perturbed_images = result["perturbed_images"]
            gen_adv_list = result["gen_adv"]
            identity_sims = result["identity_similarity"]

            # B=1 가정
            orig_tensor: torch.Tensor = original_images[0]
            pert_tensor: torch.Tensor = perturbed_images[0]
            deepfake_img = gen_adv_list[0]  # "perturbed + deepfake" 결과 (PIL)

            # PGD 후 이미지 (딥페이크 없는 버전)
            perturbed_img = _tensor_to_pil(pert_tensor)

            # |pert - orig| heatmap
            perturb_vis_img = _make_perturbation_vis(orig_tensor, pert_tensor)

            # similarity는 로그만
            identity_sim = identity_sims[0] if identity_sims else None
            if identity_sim is not None:
                logger.info(
                    f"[Job {req.jobId}] identity_similarity (orig vs gen_adv): {identity_sim:.4f}"
                )

            # 3) Spring에서 presigned URL 3종 발급
            resp_data = spring_client.request_result_upload_urls(req)
            perturbed_item = resp_data.data.perturbed
            deepfake_item = resp_data.data.deepfake
            perturb_vis_item = resp_data.data.perturbationVis

            # 4) S3 업로드 - 모두 JPEG
            # perturbed
            perturbed_bytes = _pil_to_jpeg_bytes(perturbed_img)
            perturbed_headers = _flatten_headers(perturbed_item.headers)
            logger.info(f"[Job {req.jobId}] Uploading perturbed to S3...")
            resp = requests.request(
                method=perturbed_item.method,
                url=perturbed_item.uploadUrl,
                data=perturbed_bytes,
                headers=perturbed_headers,
                timeout=60,
            )
            resp.raise_for_status()

            # deepfake (gen_adv)
            deepfake_bytes = _pil_to_jpeg_bytes(deepfake_img)
            deepfake_headers = _flatten_headers(deepfake_item.headers)
            logger.info(f"[Job {req.jobId}] Uploading deepfake to S3...")
            resp = requests.request(
                method=deepfake_item.method,
                url=deepfake_item.uploadUrl,
                data=deepfake_bytes,
                headers=deepfake_headers,
                timeout=60,
            )
            resp.raise_for_status()

            # perturbationVis
            perturb_vis_bytes = _pil_to_jpeg_bytes(perturb_vis_img)
            perturb_vis_headers = _flatten_headers(perturb_vis_item.headers)
            logger.info(f"[Job {req.jobId}] Uploading perturbationVis to S3...")
            resp = requests.request(
                method=perturb_vis_item.method,
                url=perturb_vis_item.uploadUrl,
                data=perturb_vis_bytes,
                headers=perturb_vis_headers,
                timeout=60,
            )
            resp.raise_for_status()

            # 5) complete 콜백
            spring_client.call_complete(
                req,
                perturbed_object_key=perturbed_item.objectKey,
                deepfake_object_key=deepfake_item.objectKey,
                perturbation_vis_object_key=perturb_vis_item.objectKey,
            )

            logger.info(f"[Job {req.jobId}] Completed successfully.")

        except Exception as e:
            logger.exception(f"[Job {req.jobId}] Error while processing job: {e}")
            spring_client.call_fail(req, reason="INFERENCE_ERROR")
