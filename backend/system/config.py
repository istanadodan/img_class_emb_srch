"""System Configuration - 환경 변수 및 설정"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # StudioLM Configuration
    studiolm_api_url: str = os.getenv("STUDIOLM_API_URL", "http://localhost:8000")
    studiolm_api_key: Optional[str] = os.getenv("STUDIOLM_API_KEY")

    # OpenAI Configuration (Optional)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Claude Configuration (Optional)
    claude_api_key: Optional[str] = os.getenv("CLAUDE_API_KEY")

    # Gemini Configuration (Optional)
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

    # Ollama Configuration (Optional)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Application Settings
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
