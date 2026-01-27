"""
애플리케이션 설정
환경 변수를 통해 오버라이드 가능
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings



class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # FastAPI 서버 설정
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 9000
    
    # NestJS 서버 설정
    NESTJS_URL: str = "http://localhost:3000"
    NESTJS_ANOMALY_ENDPOINT: str = "/api/v1/alert"
    NESTJS_TIMEOUT: float = 5.0
    
    # 메시지 큐 설정
    QUEUE_MAX_SIZE: int = 100
    
    # 배치 처리 설정
    INFERENCE_BATCH_SIZE: int = 64  # 한 번에 처리할 윈도우 개수
    
    # 백오프 설정
    MAX_RETRIES: int = 5
    INITIAL_DELAY: int = 1
    
    # 모델 설정
    DEFAULT_MODEL_NAME: str = "featurewise_ocsvm"  # "dummy",  "featurewise_ocsvm"
    DEFAULT_DEVICE: str = "auto"  # "auto", "cuda", "cpu"
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 전역 설정 인스턴스
settings = Settings()
