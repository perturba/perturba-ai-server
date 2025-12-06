from fastapi import FastAPI

from app.routers import jobs
from app.services.pgd_service import get_pgd_service

app = FastAPI(title="Perturba AI Server")


@app.on_event("startup")
def startup_event():
     get_pgd_service()


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(jobs.router)