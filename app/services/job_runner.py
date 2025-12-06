import logging
import threading
from io import BytesIO

import requests
from PIL import Image
import torch
from pathlib import Path

import numpy as np
from app.config import settings
from app.models.dto import FacePerturbRequest
from app.services.pgd_service import get_pgd_service
from app.clients import spring_client

logger = logging.getLogger(__name__)

DEBUG_DIR = Path("debug_outputs")
DEBUG_DIR.mkdir(exist_ok=True)

#동시에 돌아가는 Job 개수 제한
concurrency_semaphore = threading.Semaphore(settings.MAX_CONCURRENCY)


#-----------------------------------------
#presigned GET URL을 이용, 입력 이미지를 내려받음
#-----------------------------------------
def _download_image(url: str) -> Image.Image:
    logger.info(f"Downloading input image from {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


#---------------------------------------------
#PIL 이미지를 JPEG 바이트로 변환.
# - 알파채널 있으면 흰 배경으로 합성
# - mode가 RGB가 아니면 RGB로 변환
#---------------------------------------------
def _pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    # 알파 채널 제거 (RGBA, LA 등)
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])  # alpha 채널로 합성
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95, optimize=True)
    buf.seek(0)
    return buf.getvalue()

#---------------------
#JPEG 저장용: 모드 정리
#---------------------
def _save_debug_jpeg(img: Image.Image, filename: str) -> None:
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    out_path = DEBUG_DIR / filename
    img.save(out_path, format="JPEG", quality=95)
    logger.info(f"Saved debug image: {out_path}")


#--------------------------------------------------------------
#requests.request에 넣을 헤더는 str 타입이어야 함, 리스트를 문자열로 평탄화
#--------------------------------------------------------------
def _flatten_headers(headers: dict) -> dict:
    flattened = {}
    for k, v in headers.items():
        if isinstance(v, list):
            flattened[k] = ",".join(v)
        else:
            flattened[k] = str(v)
    return flattened


#------------------------------------------------------
#AI 모델이 내놓은 결과 텐서(3,H,W)를 다시 PIL 이미지로 바꾸는 함수
#PGD/SD 파이프라인에서 perturbed_images 같은 건 torch 텐서로 나올 수 있음
#presigned URL 업로드에 쓰려면 다시 PIL 이미지로 바꾼 후 PNG 바이트로 인코딩해야 함.
#------------------------------------------------------
def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    img_tensor = img_tensor.clamp(0, 1)
    np_img = (img_tensor * 255).byte().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(np_img)


#---------------------------------------------------
#원본 이미지 vs 교란된 이미지의 차이를 시각화 이미지로 만드는 함수.
#--------------------------------------------------
def _make_perturbation_vis(orig: torch.Tensor, pert: torch.Tensor) -> Image.Image:
    diff = (pert - orig).abs()
    diff_mean = diff.mean(dim=0)
    max_val = diff_mean.max()
    if max_val > 0:
        diff_mean = diff_mean / max_val
    diff_img = (diff_mean * 255).byte().cpu().numpy()
    pil = Image.fromarray(diff_img, mode="L").convert("RGB")
    return pil


#------------
#작업 처리 함수
# - 입력이미지 다운로드
# - PGD + Inpainting
# - result-upload-urls 발급
# - S3 업로드 (perturbed / deepfake / perturbationVis)
# - complete / fail 콜백
#------------
def process_face_perturb_job(req: FacePerturbRequest) -> None:
    service = get_pgd_service()

    with concurrency_semaphore:
        try:
            logger.info(f"[Job {req.jobId}] Start processing.")

            #입력 이미지 다운로드
            input_img = _download_image(str(req.inputImageUrl))

            #PGD + Inpainting 결과 얻기 (raw result 활용)
            result = service.run_attack_raw(
                input_img,
                intensity=req.intensity,
                prompt=req.prompt or "",
            )

            #result dict 구조는 pgd.py 기준 :contentReference[oaicite:1]{index=1}
            original_images = result["original_images"]
            perturbed_images = result["perturbed_images"]
            gen_adv_list = result["gen_adv"]
            gen_orig_list = result["gen_orig"]
            identity_sims = result["identity_similarity"]

            #B=1 전제
            orig_tensor: torch.Tensor = original_images[0]
            pert_tensor: torch.Tensor = perturbed_images[0]
            deepfake_img = gen_adv_list[0]
            orig_inpaint_img = gen_orig_list[0]

            #perturbed : PGD 후 이미지 (torch -> PIL)
            perturbed_img = _tensor_to_pil(pert_tensor)

            #perturbed_vis |pert - orig| heatmap
            perturb_vis_img = _make_perturbation_vis(orig_tensor, pert_tensor)

            #identity similarity는 현재 Spring에 안 보내고 로그만 남김
            identity_sim = identity_sims[0] if identity_sims else None
            if identity_sim is not None:
                logger.info(f"[Job {req.jobId}] identity_similarity (adv vs orig_inpainted): {identity_sim:.4f}")


            try:
                _save_debug_jpeg(deepfake_img, f"job-{req.jobId}-deepfake-adv.jpg")
                _save_debug_jpeg(orig_inpaint_img, f"job-{req.jobId}-deepfake-orig.jpg")
            except Exception as e:
                logger.warning(f"[Job {req.jobId}] Failed to save debug images locally: {e}")


            #스프링에서 결과 업로드용 presigned URL 발급
            resp_data = spring_client.request_result_upload_urls(req)
            perturbed_item = resp_data.data.perturbed
            deepfake_item = resp_data.data.deepfake
            perturb_vis_item = resp_data.data.perturbationVis

            #presigned URL로 업로드
            #perturbed
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

            #deepfake (gen_adv)
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

            #perturbationVis
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

            #complete 콜백
            spring_client.call_complete(
                req,
                perturbed_object_key=perturbed_item.objectKey,
                deepfake_object_key=deepfake_item.objectKey,
                perturbation_vis_object_key=perturb_vis_item.objectKey,
            )

            logger.info(f"[Job {req.jobId}] Completed successfully.")

        except Exception as e:
            logger.exception(f"[Job {req.jobId}] Error while processing job: {e}")
            #스펙상 fail API는 reason만 받으므로 간단한 코드로 넘김
            spring_client.call_fail(
                req,
                reason="INFERENCE_ERROR",
            )