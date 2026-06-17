"""Classifier Page - 이미지 분류 메인 페이지"""

import threading  # Add threading import
import streamlit as st
import asyncio
import uuid
import janus
from pathlib import Path
from datetime import datetime
from backend.service.classifier import ImageClassifierService
from backend.clients.llm_clients import StudioLLMClient
from backend.clients.embedding_client import get_studio_embedding_client as get_emb_client


# New function to run in a separate thread
def _thread_target_classification(task_id, path_input, input_type, queue_async_q):
    """Entry point for the classification thread. Runs its own asyncio event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            run_classification_task(task_id, path_input, input_type, queue_async_q)
        )
    finally:
        loop.close()


@st.fragment(run_every="1s")
def render_progress():
    """실시간 진행 상황 렌더링 (Fragment)"""
    if "active_tasks" not in st.session_state or not st.session_state.active_tasks:
        return

    st.subheader("실행 중인 작업")

    # 작업 목록 순회
    for task_id, task in list(st.session_state.active_tasks.items()):
        # janus.Queue의 sync_q를 사용하여 스레드 안전하게 읽기
        sync_q = task["queue"].sync_q

        while not sync_q.empty():
            try:
                event = sync_q.get_nowait()
                path = event.get("path")
                event_type = event.get("type")

                if event_type == "status":
                    msg = event.get("message", "")
                    task["logs"].append(f"ℹ️ {msg}")
                    # "모든 이미지 처리 완료" 메시지 수신 시 강제 카운트 보정 (안전장치)
                    if "모든 이미지 처리 완료" in msg:
                        if task["total_count"] > 0:
                            task["done_count"] = task["total_count"]

                elif event_type == "progress":
                    task["current_node"] = event.get("node", "")
                    task["current_path"] = Path(path).name if path else ""

                elif event_type in ["result", "error"]:
                    if path:
                        # 경로 정규화 (절대 경로 문자열 기반 비교)
                        norm_path: str = str(Path(path).resolve())
                        if norm_path not in task["processed_paths"]:
                            task["processed_paths"].add(norm_path)
                            task["done_count"] = len(task["processed_paths"])

                        if event_type == "result":
                            category: str = event.get("category", "no-class")
                            task["logs"].append(f"✅ 완료: {Path(path).name} -> {category}")
                        else:
                            # 개별 이미지 처리 중 에러 발생 (임베딩 실패 등)
                            task["has_error"] = True
                            node: str = event.get("node", "unknown")
                            msg: str = event.get("message", "알 수 없는 오류")
                            task["logs"].append(f"❌ 에러: {Path(path).name} ({node}) - {msg}")
                    else:
                        # 경로 정보가 없는 시스템 레벨 에러
                        if event_type == "error":
                            task["has_error"] = True
                            task["logs"].append(
                                f"⚠️ 시스템: {event.get('message', '알 수 없는 오류')}"
                            )

                elif event_type == "fatal_error":
                    task["status"] = "failed"
                    task["has_error"] = True
                    task["logs"].append(
                        f"🚨 치명적 에러: {event.get('message', '알 수 없는 오류')}"
                    )

                elif event_type == "total_count":
                    task["total_count"] = event.get("total_count", 0)

            except Exception:
                # 큐 읽기 중 개별 이벤트 처리 실패 시 무시하고 계속 진행
                continue

        # 작업 완료 상태 업데이트
        if (
            task["status"] == "running"
            and task["total_count"] > 0
            and task["done_count"] >= task["total_count"]
        ):
            task["status"] = "completed"

        # UI 출력
        status_label = task["status"].upper()
        with st.expander(
            f"Task: {task['name']} ({task['done_count']}/{task['total_count']}) - {status_label}",
            expanded=(task["status"] == "running"),
        ):
            progress = min(1.0, task["done_count"] / max(1, task["total_count"]))
            st.progress(progress)

            col1, col2 = st.columns([2, 1])
            with col1:
                if task["status"] == "running":
                    st.caption(f"현재 단계: **{task['current_node']}** - {task['current_path']}")
                elif task["status"] == "completed":
                    if task["has_error"]:
                        st.warning("작업이 완료되었으나, 일부 오류가 발생했습니다.")
                    else:
                        st.success("모든 작업이 성공적으로 완료되었습니다.")
                elif task["status"] == "failed":
                    st.error("작업 중 치명적인 오류가 발생했습니다.")

                if task["status"] in ["completed", "failed"]:
                    if st.button("목록에서 제거", key=f"del_{task_id}"):
                        del st.session_state.active_tasks[task_id]
                        st.rerun()

            with col2:
                st.caption(f"시작: {task['start_time']}")

            # 모든 로그 표시
            for log in task["logs"]:
                st.text(log)


async def run_classification_task(task_id: str, path_input: str, input_type: str, queue_async_q: Any):
    """백그라운드 분류 작업 실행 (비차단)"""
    service: ImageClassifierService = ImageClassifierService(
        llm_client=StudioLLMClient(), embedding_client=get_emb_client()
    )
    
    # 1. 경로 정규화: 공백 제거, 따옴표 제거, 역슬래시를 슬래시로 변환(도커 호환성)
    clean_path: str = path_input.strip().replace('"', "").replace("'", "")
    
    # 2. 경로 객체 생성 및 절대 경로 여부 확인
    p: Path = Path(clean_path)
    
    # 도커 환경(/app)에서 입력한 경로가 절대 경로인 경우 그대로 사용, 
    # 상대 경로인 경우에만 현재 디렉토리 기준으로 resolve
    final_path: str = str(p.absolute()) if p.is_absolute() else str(p.resolve())

    try:
        if input_type == "📁 폴더":
            await service.classify_folder(folder_path=final_path, event_queue=queue_async_q)
        else:
            await service.classify_batch([final_path], event_queue=queue_async_q)

    except Exception as e:
        await queue_async_q.put({"type": "fatal_error", "message": f"치명적 오류: {str(e)}"})
    finally:
        pass


async def render_classifier_page():
    """이미지 분류 페이지 렌더링"""

    st.title("🖼️ 멀티 태스크 이미지 분류")
    st.write("여러 작업을 동시에 실행하고 실시간으로 진행 상황을 모니터링할 수 있습니다.")

    # 세션 상태 초기화
    if "active_tasks" not in st.session_state:
        st.session_state.active_tasks = {}

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            input_type = st.radio("입력 방식 선택", ["📁 폴더", "🖼️ 이미지 파일"], horizontal=True)

        with col2:
            if input_type == "📁 폴더":
                path_input = st.text_input(
                    "폴더 경로",
                    placeholder="예: C:\\Users\\Pictures\\MyPhotos",
                )
            else:
                path_input = st.text_input(
                    "이미지 파일 경로",
                    placeholder="예: C:\\Users\\Pictures\\photo.jpg",
                )

    if st.button("🚀 신규 분류 시작", width="stretch", type="primary"):
        if not path_input:
            st.error("경로를 입력해주세요.")
        else:
            # janus.Queue 생성 (Thread-safe)
            queue = janus.Queue()
            task_id = str(uuid.uuid4())
            task_name = Path(path_input).name or path_input

            st.session_state.active_tasks[task_id] = {
                "name": task_name,
                "queue": queue,
                "status": "running",
                "total_count": 0,
                "done_count": 0,
                "processed_paths": set(),  # Add for duplicate count prevention
                "has_error": False,  # Add for error status tracking
                "current_node": "준비 중",
                "current_path": "",
                "logs": [],
                "start_time": datetime.now().strftime("%H:%M:%S"),
            }

            # 백그라운드 태스크 시작 (새로운 스레드에서 실행)
            thread = threading.Thread(
                target=_thread_target_classification,
                args=(task_id, path_input, input_type, queue.async_q),
            )
            thread.daemon = True  # Allow program to exit even if thread is still running
            thread.start()

            st.toast(f"작업 시작: {task_name}")
            st.rerun()

    # 진행 상황 표시 (Fragment)
    render_progress()
