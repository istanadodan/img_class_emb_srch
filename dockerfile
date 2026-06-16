# 1. 가벼운 Python Slim 이미지 사용
FROM python:3.14.2-slim

# 2. 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 의존성 설치 (Streamlit 및 포트 활성화를 위한 기본 도구)
# curl은 컨테이너 상태 체크용(선택)이며, 무거워지지 않도록 캐시를 삭제합니다.
RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y \
curl \
&& rm -rf /var/lib/apt/lists/*

# 4. 종속성 파일 복사 및 설치
# 의존성 설치를 먼저 해야 소스코드가 바뀔 때마다 패키지를 새로 설치하지 않습니다 (캐시 활용)
COPY requirements.txt .
RUN pip install -r requirements.txt
# --no-cache-dir \
# --extra-index-url https://download.pytorch.org/whl/cu126 \

# 5. 프로젝트 소스 코드 복사
COPY . .

# 6. Streamlit이 사용하는 기본 포트 노출
EXPOSE 8501

# 7. Streamlit 실행 명령어
ENTRYPOINT ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]