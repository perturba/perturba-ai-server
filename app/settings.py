from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "perturba-ai"
    ENV: str = "local"

    BACKEND_BASE_URL: str = "https://your-backend.example.com"
    BACKEND_API_TOKEN: str = "dev-secret"

    PRESIGN_PATH_TMPL: str = "/v1/jobs/{publicId}/presign-outputs"

    COMPLETE_PATH_TMPL: str = "/v1/jobs/{publicId}/complete"
    FAIL_PATH_TMPL: str = "/v1/jobs/{publicId}/fail"

    SIMULATE_SEC: float = 2.0
    HTTP_TIMEOUT_SEC: float = 20.0

    class Config:
        env_file = ".env"

settings = Settings()
