import logging
from typing import Tuple
import requests
from app.config import settings
from app.models.dto import (
    FacePerturbRequest,
    ApiResponseResultUploadUrlResponse,
    CompleteResultRequest,
    JobFailRequest,
)

logger = logging.getLogger(__name__)
JOBS_BASE = f"{settings.API_BASE_URL}/v1/internal/jobs"


#----------------------------------------------------
# POST /v1/internal/jobs/{jobId}/result-upload-urls
# - request body 없음
# - 응답: ApiResponseResultUploadUrlResponse
# 작업 완료 후 이미지 업로드를 위한 PUt Presigned URL 받아옴
#----------------------------------------------------
def request_result_upload_urls(
        req: FacePerturbRequest,
) -> ApiResponseResultUploadUrlResponse:
    url = f"{JOBS_BASE}/{req.jobId}/result-upload-urls"

    logger.info(f"[Job {req.jobId}] Requesting result-upload-urls from Spring: {url}")
    resp = requests.post(url, timeout=30)
    resp.raise_for_status()

    data = ApiResponseResultUploadUrlResponse(**resp.json())
    if not data.ok:
        raise RuntimeError("Spring responded with ok=false for result-upload-urls")

    return data


#-----------------------------------------
# POST /v1/internal/jobs/{jobId}/complete
# Body: CompleteResultRequest
# 작업 완료 + 이미지 업로드 완료를 백엔드에 알림
#-----------------------------------------
def call_complete(
    req: FacePerturbRequest,
    perturbed_object_key: str,
    deepfake_object_key: str,
    perturbation_vis_object_key: str,
) -> None:
    url = f"{JOBS_BASE}/{req.jobId}/complete"

    payload = CompleteResultRequest(
        perturbedObjectKey=perturbed_object_key,
        deepfakeObjectKey=deepfake_object_key,
        perturbationVisObjectKey=perturbation_vis_object_key,
    ).dict()

    logger.info(f"[Job {req.jobId}] Calling complete callback: {url}")
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()


#-------------------------------------------
# POST /v1/internal/jobs/{publicId}/fail
# Body: JobFailRequest {reason}
# 작업 진행중 오류발생으로 인한 call back
#--------------------------------------------
def call_fail(req: FacePerturbRequest, reason: str) -> None:

    url = f"{JOBS_BASE}/{req.publicId}/fail"

    payload = JobFailRequest(reason=reason).dict()

    logger.info(f"[Job {req.jobId}] Calling fail callback: {url}")
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"[Job {req.jobId}] Fail callback itself failed: {e}")