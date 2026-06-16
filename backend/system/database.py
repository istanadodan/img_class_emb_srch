"""Database Configuration - SQLAlchemy PostgreSQL 설정"""

import inspect
from functools import wraps
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.system.config import settings

# PostgreSQL 연결 문자열
DATABASE_URL = (
    f"postgresql+psycopg2://{settings.db_user}:{settings.db_password}"
    f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
)

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
)


# pgvector 확장 활성화
@event.listens_for(engine, "connect")
def on_connect(dbapi_conn, connection_record):
    """PostgreSQL 연결 시 pgvector 확장 활성화"""
    with dbapi_conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        dbapi_conn.commit()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """데이터베이스 세션 반환"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


from sqlalchemy.exc import OperationalError
from backend.system.exceptions import DatabaseConnectionException


def init_db():
    """데이터베이스 테이블 생성"""
    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError as e:
        raise DatabaseConnectionException(f"Failed to connect to database during initialization: {str(e)}")


@contextmanager
def _transactional_session(func, kwargs):
    """트랜잭션 세션 생명주기 관리를 위한 공통 컨텍스트 매니저"""
    db_session = None
    # 'db'가 인자에 없고 함수가 'db'를 요구하는 경우에만 세션 생성
    if "db" not in kwargs:
        sig = inspect.signature(func)
        if "db" in sig.parameters:
            try:
                db_session = SessionLocal()
                # 실제 연결 확인을 위해 간단한 쿼리 실행 시도 가능하나 오버헤드 고려
                kwargs["db"] = db_session
            except OperationalError as e:
                raise DatabaseConnectionException(f"Database connection failed: {str(e)}")

    try:
        yield
        if db_session:
            db_session.commit()
    except OperationalError as e:
        if db_session:
            db_session.rollback()
        raise DatabaseConnectionException(f"Database operation failed due to connection error: {str(e)}")
    except Exception:
        if db_session:
            db_session.rollback()
        raise
    finally:
        if db_session:
            db_session.close()


def transactional(func):
    """
    데이터베이스 트랜잭션을 관리하는 데코레이터.
    동기(sync) 및 비동기(async) 함수를 모두 지원하며 세션을 자동 주입합니다.
    """

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        with _transactional_session(func, kwargs):
            return func(*args, **kwargs)

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        with _transactional_session(func, kwargs):
            return await func(*args, **kwargs)

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
