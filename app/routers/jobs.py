import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.models.dto import FacePerturbRequest, AcceptedResponse
from app.services.pgd_service import get_pgd_service
from app.services.job_runner import process_face_perturb_job

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/internal/ai",
    tags=["ai"],
)


#--------------------------------------------
# Spring Boot가 호출하는 엔트리 API.
# - 요청 검증 후 BackgroundTasks에 실제 작업을 던지고 바로 accepted 응답 리턴
#--------------------------------------------
@router.post("/face-perturb", response_model=AcceptedResponse)
def face_perturb(
    body: FacePerturbRequest,
    background_tasks: BackgroundTasks,
):
    service = get_pgd_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info(f"Received face-perturb request for jobId={body.jobId}")
    background_tasks.add_task(process_face_perturb_job, body)

    return AcceptedResponse()