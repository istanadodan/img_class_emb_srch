"""System Exceptions - 시스템 전체에서 사용하는 예외"""


class PicRecogException(Exception):
    """PicRecog 기본 예외"""

    pass


class AIClientException(PicRecogException):
    """AI 클라이언트 관련 예외"""

    pass


class ImageProcessingException(PicRecogException):
    """이미지 처리 관련 예외"""

    pass


class ConfigurationException(PicRecogException):
    """설정 관련 예외"""

    pass


class DatabaseConnectionException(PicRecogException):
    """데이터베이스 연결 실패 관련 예외"""

    pass
