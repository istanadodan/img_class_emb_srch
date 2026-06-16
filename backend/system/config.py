"""System Configuration - 환경 변수 및 설정"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import logging
import sys


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # ... (existing fields)
    studiolm_api_url: str = Field(default="http://localhost:11434")
    studiolm_api_key: str = Field(default="lm-studio")
    embedding_model_name: str = Field(default="Qwen/Qwen3-VL-Embedding-2B")
    llm_model_name: str = Field(default="Qwen/Qwen3.5-9B")

    # Database Configuration (PostgreSQL)
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=41659)
    db_name: str = Field(default="imgdb")
    db_user: str = Field(default="postgres")
    db_password: str = Field(default="postgres")

    # Application Settings
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def setup_logging(self):
        """프로젝트 로깅 설정"""
        level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
            force=True,
        )
        # 타 라이브러리 로그 레벨 조정
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("pydantic").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers.utils.doc").setLevel(logging.ERROR)


settings = Settings()
