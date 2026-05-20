# PicRecog - AI 기반 이미지 자동 분류 시스템

Streamlit UI와 LangGraph 기반 Backend로 구성된 현대적인 이미지 분류 시스템입니다.

이미지 경로/폴더를 입력받아 AI 모델로 4가지 카테고리(인물, 꽃/경치, 글자, 행사)로 자동 분류합니다.

## 프로젝트 구조

```
picRecog/
├── backend/                  # 백엔드 (LangGraph 최적화)
│   ├── domain/              # 도메인 객체, 엔티티
│   │   └── models.py        # ImageCategory, ClassificationResult
│   ├── system/              # 로깅, 설정, 예외처리
│   │   ├── config.py        # 환경 설정
│   │   └── exceptions.py    # 커스텀 예외
│   ├── service/             # 비즈니스 로직
│   │   └── classifier.py    # 이미지 분류 서비스
│   ├── presentation/        # 응답 스키마
│   │   └── schemas.py       # API 응답 형식
│   ├── clients/             # AI 모델 클라이언트
│   │   ├── base.py          # 기본 인터페이스
│   │   └── studiolm.py      # StudioLM API 클라이언트
│   └── graph/               # LangGraph 워크플로우
│       └── workflow.py      # 분류 워크플로우
├── ui/                       # 프론트엔드 (Streamlit)
│   ├── app.py              # 메인 애플리케이션
│   ├── pages/              # 페이지들
│   │   └── classifier.py   # 분류 페이지
│   ├── components/         # 재사용 컴포넌트
│   │   └── result_display.py
│   └── styles/             # 테마, 스타일
│       └── theme.py
├── shared/                  # 공유 모듈
│   └── constants.py        # 상수 정의
├── tests/                   # 테스트
├── pyproject.toml          # 프로젝트 설정
├── .env.example            # 환경 변수 템플릿
└── CLAUDE.md               # 개발 가이드
```

## 설치

### 1. uv 설치

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 의존성 설치

```bash
# 프로젝트 의존성 설치
uv sync --all-extras
```

### 3. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# StudioLM API 설정 (필수)
# STUDIOLM_API_URL=http://localhost:8000
# STUDIOLM_API_KEY=your_api_key
```

## 실행

### Streamlit UI 실행

```bash
uv run streamlit run ui/app.py
```

UI는 `http://localhost:8501`에서 접속 가능합니다.

## 개발

### 코드 품질

```bash
# 포맷팅
uv run black backend/ ui/ shared/ tests/

# 린팅
uv run ruff check backend/ ui/ shared/ tests/
uv run ruff format backend/ ui/ shared/ tests/

# 타입 체킹
uv run mypy backend/ ui/ shared/
```

### 테스트

```bash
# 전체 테스트
uv run pytest tests/ -v

# 커버리지 포함
uv run pytest tests/ --cov=backend --cov=ui --cov=shared
```

## 아키텍처

### 계층 분리

- **Domain**: 도메인 엔티티, 비즈니스 규칙
- **System**: 공통 유틸, 설정, 로깅
- **Service**: 비즈니스 로직
- **Presentation**: API 스키마
- **Clients**: AI 모델 클라이언트 (추상화)
- **Graph**: LangGraph 워크플로우

### AI 클라이언트 패턴

모든 AI 제공자는 `AIClient` 인터페이스를 구현합니다:

```python
class AIClient(ABC):
    async def classify_image(self, image_path: str) -> dict:
        """이미지 분류"""
        pass

    def is_available(self) -> bool:
        """사용 가능 여부 확인"""
        pass
```

지원되는 클라이언트:
- StudioLM (Primary)
- OpenAI
- Claude
- Gemini
- Ollama

## 환경 변수

```bash
# StudioLM
STUDIOLM_API_URL=http://localhost:8000
STUDIOLM_API_KEY=...

# Optional: 다른 AI 제공자
OPENAI_API_KEY=...
CLAUDE_API_KEY=...
GEMINI_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434

# Application
DEBUG=False
LOG_LEVEL=INFO
```

## 리소스

- [CLAUDE.md](CLAUDE.md) - 개발 가이드
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 라이센스

MIT
