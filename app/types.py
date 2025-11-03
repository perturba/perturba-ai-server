from pydantic import BaseModel, HttpUrl
from typing import Literal, Dict, Any

class JobRequest(BaseModel):
    public_id: str
    input_url: HttpUrl
    intensity: Literal["LOW","MEDIUM","HIGH"] = "MEDIUM"
    model: str = "perturba-v1"

class PresignResponse(BaseModel):
    perturbed: Dict[str, Any]          # { put_url, headers, public_url }
    deepfake_output: Dict[str, Any]    # { put_url, headers, public_url }
    perturbation_vis: Dict[str, Any]   # { put_url, headers, public_url }

class CompletePayload(BaseModel):
    perturbedUrl: HttpUrl
    deepfakeUrl: HttpUrl
    perturbationVisUrl: HttpUrl

class FailPayload(BaseModel):
    reason: str
