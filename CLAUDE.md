# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PicRecog**: 이미지 경로/폴더를 입력받아 AI 모델로 자동 분류하고 Streamlit UI로 시각화하는 시스템.

**Image Categories** (4가지):
1. 인물 (People)
2. 꽃 또는 경치 (Flowers/Landscape)
3. 글자 등 문자포함 (Text/Documents)
4. 행사 (Events: 여행, 생일, 산책 등)

**Tech Stack**:
- Backend: LangGraph + FastAPI (또는 Flask)
- UI: Streamlit
- AI Models: StudioLM API (Primary) + OpenAI, Claude, Gemini, Ollama 지원
- Package Manager: uv

## Project Structure

```
picRecog/
├── backend/
│   ├── domain/              # 도메인 객체, 엔티티, DTO
│   ├── system/              # 로깅, 설정, 예외처리 등
│   ├── service/             # 비즈니스 로직 (이미지 분류)
│   ├── presentation/        # 응답 스키마, 형식
│   ├── clients/             # AI 모델 클라이언트 추상화
│   │   ├── base.py          # 기본 클라이언트 인터페이스
│   │   ├── studiolm.py      # StudioLM API 클라이언트
│   │   ├── openai_client.py
│   │   └── ...
│   ├── graph/               # LangGraph 워크플로우
│   └── main.py
├── ui/
│   ├── pages/               # Streamlit 페이지
│   │   └── classifier.py    # 메인 분류 페이지
│   ├── components/          # 재사용 컴포넌트
│   ├── styles/              # CSS/스타일 설정
│   └── app.py               # Streamlit 진입점
├── shared/                  # 공유 모듈 (models, exceptions)
├── tests/
├── pyproject.toml
└── .env.example
```

**계층 구조 원칙**:
- 최대 5단계 금지, 가급적 3단계 내
- 도메인과 시스템 공통요소 분리
- 서비스와 프레젠테이션 계층 명확 분리

## Development Commands

Uses `uv` as the package manager.

### Setup
```bash
# Install dependencies and create virtual environment
uv sync --all-extras
```

### Running
```bash
# Run Streamlit UI
uv run streamlit run ui/app.py

# Run backend tests
uv run pytest tests/ -v
```

### Code Quality
```bash
# Format
uv run black backend/ ui/ shared/ tests/

# Lint and check
uv run ruff check backend/ ui/ shared/ tests/
uv run ruff format backend/ ui/ shared/ tests/

# Type checking
uv run mypy backend/ ui/ shared/

# Tests with coverage
uv run pytest tests/ --cov=backend --cov=ui --cov=shared
```

## AI Client Pattern

모든 AI 클라이언트는 공통 인터페이스를 구현:

```python
from abc import ABC, abstractmethod
from typing import Literal

ImageCategory = Literal["people", "nature", "text", "events"]

class AIClient(ABC):
    @abstractmethod
    async def classify_image(self, image_path: str) -> ImageCategory:
        """이미지를 분류하고 카테고리 반환"""
        pass
```

각 제공자별 클라이언트 구현:
- `StudioLMClient`: Local StudioLM API
- `OpenAIClient`: OpenAI Vision API
- `ClaudeClient`: Claude Vision API
- `GeminiClient`: Google Gemini API
- `OllamaClient`: Local Ollama

## LangGraph Workflow

이미지 분류 워크플로우 (참고):

```
START → Load Image → Classify with AI → Validate Result → END
```

State 정의:
- `image_path`: 입력 이미지 경로
- `category`: 분류 결과
- `confidence`: 신뢰도
- `metadata`: 추가 정보

## UI Components (Streamlit)

- **File Uploader**: 이미지/폴더 입력
- **Progress Bar**: 분류 진행 상황
- **Results Display**: 카테고리별 이미지 그룹화 표시
- **Modern Styling**: Streamlit theming + CSS

## Environment Variables

`.env` 파일 구성:
```
# StudioLM (Primary)
STUDIOLM_API_URL=http://localhost:8000
STUDIOLM_API_KEY=your_key

# Optional fallbacks
OPENAI_API_KEY=...
CLAUDE_API_KEY=...
GEMINI_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434
```

## Important Notes

- **Client 추상화**: 향후 다양한 AI 모델 추가 용이하도록 설계
- **계층 분리**: 도메인/서비스/프레젠테이션 명확히
- **UI/Backend 독립**: 별도 패키지로 독립적 개발/배포 가능
- **추가 요구사항**: 이후 별도 지시
