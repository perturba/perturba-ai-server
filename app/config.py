import os


class Settings:
    #동시에 몇 개까지 AI 작업을 돌릴지?
    MAX_CONCURRENCY: int = int(os.getenv("MAX_CONCURRENCY", "1"))


settings = Settings()
