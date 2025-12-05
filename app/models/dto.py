from typing import Optional
from pydantic import BaseModel, HttpUrl


#-------------------------------
#API Server으로부터 받는 작업 요청 dto
#-------------------------------
class FacePerturbRequest(BaseModel):
    jobId: int
    publicId: str
    inputImageUrl: HttpUrl  #이미지 url
    prompt: Optional[str] = ""  #stable diffusion prompt (없으면 "")


#response
class AcceptedResponse(BaseModel):
    status: str = "accepted"


#-----------------------------------------------------------------
#/v1/internal/jobs/{jobId}/result-upload-urls 요청 보내고 받는 응답 Dto
#-----------------------------------------------------------------
class UploadItem(BaseModel):
    method: str
    uploadUrl: str  #presigned URL
    headers: dict[str, list[str]]  #{ "Content-Type": ["image/png"], ... }
    objectKey: str  #S3 objectKey


class ResultUploadUrlData(BaseModel):
    perturbed: UploadItem
    deepfake: UploadItem
    perturbationVis: UploadItem


#이렇게 응답이 올 것
class ApiResponseResultUploadUrlResponse(BaseModel):
    ok: bool
    data: ResultUploadUrlData

    class Config:
        extra = "ignore"


#--------------------------------------------------
# /v1/internal/jobs/{jobId}/complete에 보내는 요청 dto
#--------------------------------------------------
class CompleteResultRequest(BaseModel):
    perturbedObjectKey: str
    deepfakeObjectKey: str
    perturbationVisObjectKey: str


#------------------------------------------------
# /v1/internal/jobs/{publicId}/fail에 보내는 요청 dto
#-------------------------------------------------
class JobFailRequest(BaseModel):
    reason: str
