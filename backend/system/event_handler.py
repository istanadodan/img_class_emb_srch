import asyncio
import logging
from typing import Optional, Any, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class UIEventHandler:
    """UI 실시간 피드백을 위한 이벤트 전송 공통 클래스"""
    
    @staticmethod
    async def publish(
        event_queue: Optional[Any],
        event_type: str,
        message: str,
        path: Optional[str] = None,
        node: Optional[str] = None,
        **extra
    ):
        """UI 이벤트를 큐에 발행 (비동기) - janus.Queue.async_q 호환"""
        if not event_queue:
            return

        payload = {
            "type": event_type,
            "message": message,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **extra
        }
        if path: payload["path"] = path
        if node: payload["node"] = node
        
        # janus.Queue의 경우 .async_q 속성이 있으면 그것을 사용
        target_q = getattr(event_queue, "async_q", event_queue)
        await target_q.put(payload)

class WorkflowLogger:
    """워크플로우 상태 기록을 위한 로깅 공통 클래스"""
    
    @staticmethod
    def log(node: str, message: str, level: str = "info", **extra):
        """표준 파이썬 로깅 출력"""
        log_msg = f"[{node}] {message}"
        if level == "error":
            logger.error(log_msg)
        elif level == "warning":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    @staticmethod
    def make_state_log(node: str, message: str, **extra) -> dict:
        """ClassificationState의 logs 리스트에 저장할 딕셔너리 생성"""
        return {
            "node": node,
            "message": message,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **extra
        }
